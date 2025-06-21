//
//  NotesConverter.swift
//  ap
//
//  Created by 张宇飞 on 2024/11/22.
//

import Accelerate
import CoreML

private typealias InnerNote = (Int, Int, Int, Float)

private func hzToMidi(_ freq: Float) -> Int {
    return Int((12 * (log2(freq) - log2(440.0)) + 69).rounded())
}

private func midiToHz(_ pitch: Int) -> Float {
    return pow(Float(2.0), Float(pitch - 69) / Float(12.0)) * Float(440.0)
}

private func constrainFrequency(onsets: MLMultiArray, frames: MLMultiArray, maxFreq: Float?, minFreq: Float?) -> (
    MLMultiArray,
    MLMultiArray
) {
    if maxFreq != nil {
        let limitMax: (_ arr: MLMultiArray, _ pitch: Int) -> Void = { arr, pitch in
            var len = vDSP_Length(arr.shape[0].intValue)
            var limit = arr.shape[1].intValue
            if pitch <= limit {
                arr.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, strides in
                    for i in pitch..<limit {
                        vDSP_vclr(ptr.baseAddress!.advanced(by: i), strides[0], len)
                    }
                }
            }
        }

        let pitch = hzToMidi(maxFreq!) - Constants.midiOffset
        limitMax(onsets, pitch)
        limitMax(frames, pitch)
    }

    if minFreq != nil {
        let limitMin: (_ arr: MLMultiArray, _ pitch: Int) -> Void = { arr, pitch in
            var len = vDSP_Length(arr.shape[0].intValue)
            var limit = arr.shape[1].intValue
            if pitch >= 0 {
                arr.withUnsafeMutableBufferPointer(ofType: Float.self) { ptr, strides in
                    for i in 0..<pitch {
                        vDSP_vclr(ptr.baseAddress!.advanced(by: i), strides[0], len)
                    }
                }
            }
        }

        let pitch = hzToMidi(minFreq!) - Constants.midiOffset
        limitMin(onsets, pitch)
        limitMin(frames, pitch)
    }
    return (onsets, frames)
}

/// 根据振幅出现较大变化的特征位置推断 onsets
private func getInferedOnsets(onsets: MLMultiArray, frames: MLMultiArray, nDiff: Int = 2) -> MLMultiArray? {
    // 获取差异
    var diffs: [[Float]] = []
    let frame1Size = frames.shape[1].intValue
    frames.withUnsafeBufferPointer(ofType: Float.self) { ptr in
        for i in 1...nDiff {
            let tmpFrames = [Float](repeating: 0, count: frame1Size * i) + ptr
            diffs.append(vDSP.subtract(ptr, tmpFrames[..<ptr.count]))
        }
    }

    // 求最小值 frame_diff = np.min(diffs, axis = 0)
    var frameDiff = diffs[0]
    for i in 0..<(nDiff - 1) {
        vDSP_vmin(diffs[i], 1, diffs[i + 1], 1, &frameDiff, 1, vDSP_Length(frameDiff.count))
    }

    // 设定阈值 frame_diff[frame_diff < 0] = 0
    vDSP_vthres(frameDiff, 1, [0], &frameDiff, 1, vDSP_Length(frameDiff.count))

    // frame_diff[:n_diff, :] = 0
    vDSP_vclr(&frameDiff, 1, vDSP_Length(nDiff * frame1Size))

    // frame_diff = np.max(onsets) * frame_diff / np.max(frame_diff)  # rescale to have the same max as onsets
    let ret = try? MLMultiArray(shape: onsets.shape, dataType: onsets.dataType)
    ret?.withUnsafeMutableBufferPointer(ofType: Float.self) { dptr, _ in
        onsets.withUnsafeBufferPointer(ofType: Float.self) { ptr in
            let maxDiff = vDSP.maximum(frameDiff)
            var i = vDSP.maximum(ptr) / (maxDiff == 0 ? 1 : maxDiff)
            vDSP_vsmul(frameDiff, 1, &i, &frameDiff, 1, vDSP_Length(frameDiff.count))

            // max_onsets_diff = np.max([onsets, frame_diff], axis=0)  # use the max of the predicted onsets and the differences
            vDSP_vmax(frameDiff, 1, ptr.baseAddress!, 1, dptr.baseAddress!, 1, vDSP_Length(frameDiff.count))
        }
    }
    return ret
}

private func findValidOnsetIdxsTurbo(onsets: MLMultiArray, thresh: Float) -> [(Int, Int)] {
    let shape = onsets.shape.map { v in
        return v.intValue
    }
    // FIXME: 至少 3 个才能组成极值查找（不足 3 的情况应该也需要处理)
    if shape[0] < 3 {
        return []
    }
    let step = onsets.strides[0].intValue
    return onsets.withUnsafeBufferPointer(ofType: Float.self) { ptr in
        // 生成一个索引矩阵 indices 保存每个数字的位置
        var indices = vDSP.ramp(withInitialValue: Float(0), increment: Float(1), count: ptr.count)
        // 过滤数据，把小于 thresh 的都设为 0，生成矩阵 gating
        var gating = vDSP.ramp(withInitialValue: Float(0), increment: Float(0), count: ptr.count)
        vDSP_vthres(ptr.baseAddress!, 1, [thresh], &gating, 1, vDSP_Length(ptr.count))
        // 压缩后获得 >= thresh 的位置列表
        indices = vDSP.compress(indices, gatingVector: gating, nonZeroGatingCount: nil)

        // 判定索引数组中有效的位置即可
        var row: Int = 0
        var idx: Int = 0
        var v: Float = 0
        let rowLimit = shape[0] - 1
        var ret: [(Int, Int)] = []
        for i in indices {
            idx = Int(i)
            row = idx / step
            if row == 0 || row == rowLimit {
                continue
            }
            v = ptr[idx]
            if v > ptr[idx - step] && v > ptr[idx + step] {
                ret.append((row, idx % step))
            }
        }
        return ret
    }
}

/// 将原始模型输出转换成复音音符事件
private func outputToNotesPolyphonic(
    frames: MLMultiArray, onsets: MLMultiArray, onsetThresh: Float, frameThresh: Float, minNoteLen: Int,
    inferOnsets: Bool, maxFreq: Float?, minFreq: Float?,
    melodiaTrick: Bool, energyTol: Int
) -> [InnerNote] {
    let nFrames = frames.shape[0].intValue
    var (onsets, frames) = constrainFrequency(onsets: onsets, frames: frames, maxFreq: maxFreq, minFreq: minFreq)
    if inferOnsets {
        onsets = getInferedOnsets(onsets: onsets, frames: frames)!
    }

    let frameStep = frames.strides[0].intValue
    return frames.withUnsafeBufferPointer(ofType: Float.self) { pFrames in
        var remainingEnergy = Array(pFrames)
        var noteEvents: [InnerNote] = []
        var onsetIdxs = findValidOnsetIdxsTurbo(onsets: onsets, thresh: onsetThresh)
        onsetIdxs.reverse()

        for (noteStartIdx, freqIdx) in onsetIdxs {
            if noteStartIdx >= nFrames - 1 {
                continue
            }

            var i = noteStartIdx + 1
            var k = 0
            while i < nFrames - 1 && k < energyTol {
                if remainingEnergy[i * frameStep + freqIdx] < frameThresh {
                    k += 1
                } else {
                    k = 0
                }
                i += 1
            }

            i -= k

            if i - noteStartIdx <= minNoteLen {
                continue
            }

            let startPos = noteStartIdx * frameStep + freqIdx
            let clrLen = vDSP_Length(i - noteStartIdx)
            vDSP_vclr(&remainingEnergy[startPos], frameStep, clrLen)
            if freqIdx < Constants.maxFreqIdx {
                vDSP_vclr(&remainingEnergy[startPos + 1], frameStep, clrLen)
            }
            if freqIdx > 0 {
                vDSP_vclr(&remainingEnergy[startPos - 1], frameStep, clrLen)
            }

            var amplitude: Float = 0
            vDSP_meanv(pFrames.baseAddress!.advanced(by: startPos), frameStep, &amplitude, clrLen)
            noteEvents.append((noteStartIdx, i, freqIdx + Constants.midiOffset, amplitude))
        }

        if melodiaTrick {
            var maxValue: Float = 0
            var maxIdx: vDSP_Length = 0
            var amplitude: Float = 0
            let limit = vDSP_Length(remainingEnergy.count)
            var i = 0
            var k = 0
            var startPos = 0

            while true {
                vDSP_maxvi(remainingEnergy, 1, &maxValue, &maxIdx, limit)
                if maxValue <= frameThresh {
                    break
                }

                let iMid = Int(maxIdx) / frameStep
                let freqIdx = Int(maxIdx) % frameStep
                remainingEnergy[iMid * frameStep + freqIdx] = 0

                i = iMid + 1
                k = 0
                while i < nFrames - 1 && k < energyTol {
                    startPos = i * frameStep + freqIdx
                    if remainingEnergy[startPos] < frameThresh {
                        k += 1
                    } else {
                        k = 0
                    }
                    remainingEnergy[startPos] = 0
                    if freqIdx < Constants.maxFreqIdx {
                        remainingEnergy[startPos + 1] = 0
                    }
                    if freqIdx > 0 {
                        remainingEnergy[startPos - 1] = 0
                    }
                    i += 1
                }
                let iEnd = i - 1 - k

                i = iMid - 1
                k = 0
                while i > 0 && k < energyTol {
                    startPos = i * frameStep + freqIdx
                    if remainingEnergy[startPos] < frameThresh {
                        k += 1
                    } else {
                        k = 0
                    }
                    remainingEnergy[startPos] = 0
                    if freqIdx < Constants.maxFreqIdx {
                        remainingEnergy[startPos + 1] = 0
                    }
                    if freqIdx > 0 {
                        remainingEnergy[startPos - 1] = 0
                    }
                    i -= 1
                }
                let iStart = i + 1 + k
                assert(iStart >= 0, "\(iStart)")
                assert(iEnd < nFrames)

                let iLen = iEnd - iStart
                if iLen <= minNoteLen {
                    continue
                }

                vDSP_meanv(
                    pFrames.baseAddress!.advanced(by: iStart * frameStep + freqIdx),
                    frameStep,
                    &amplitude,
                    vDSP_Length(iLen)
                )
                noteEvents.append((iStart, iEnd, freqIdx + Constants.midiOffset, amplitude))
            }
        }
        return noteEvents
    }
}

private func framesToTime(_ frames: [Float], sr: Int, hopLength: Int) -> [Float] {
    // frames to sample
    var samples = vDSP.multiply(Float(hopLength), frames)
    vForce.floor(samples, result: &samples)
    // samples to time
    vDSP.divide(samples, Float(sr), result: &samples)
    return samples
}

private func modelFrameToTime(_ nFrames: Int) -> [Float] {
    var frames = vDSP.ramp(withInitialValue: Float(0), increment: Float(1), count: nFrames)
    let oriTimes = framesToTime(frames, sr: Constants.audioSampleRate, hopLength: Constants.fftHop)
    vDSP.divide(frames, Float(Constants.annotNFrames), result: &frames)
    vForce.floor(frames, result: &frames)

    let windowOffset =
        Float(Constants.fftHop) / Float(Constants.audioSampleRate)
        * (Float(Constants.annotNFrames) - (Float(Constants.audioNSamples) / Float(Constants.fftHop))) + Float(0.0018)

    vDSP.multiply(windowOffset, frames, result: &frames)
    vDSP.subtract(oriTimes, frames, result: &frames)
    return frames
}

// 生成高斯信号窗口 scipy.signal.windows.gaussian
private func makeWindowsGaussian(_ count: Int, std: Int) -> [Float] {
    if count <= 1 {
        return [Float](repeating: Float(1), count: count)
    }
    // n = np.arange(0, M) - (M - 1.0) / 2.0
    var n = vDSP.ramp(withInitialValue: Float(count - 1) * Float(-0.5), increment: Float(1), count: count)
    let sig2 = Float(2 * std * std)
    // w = np.exp(-n ** 2 / sig2)
    vDSP.square(n, result: &n)
    vDSP.divide(n, -sig2, result: &n)
    vForce.exp(n, result: &n)

    return n
}

private func midiPitchToContourBin(_ pitch: Int) -> Float {
    let hz = midiToHz(pitch)
    return Float(12.0) * Float(Constants.contoursBinsPerSemitone) * log2(hz / Constants.annotationsBaseFrequency)
}

private func getPitchBends(contours: MLMultiArray, noteEvents: [InnerNote], nBinsTolerance: Int = 25) -> [(
    InnerNote, PitchBend?
)] {
    let contourCols = contours.strides[0].intValue
    return contours.withUnsafeBufferPointer(ofType: Float.self) { pcontours in
        let windowLen = nBinsTolerance * 2 + 1
        let freqGaussian = makeWindowsGaussian(windowLen, std: 5)
        return freqGaussian.withUnsafeBufferPointer { pgaussian in
            var freqIdx: Int
            var freqStartIdx: Int
            var freqEndIdx: Int
            var gaussianIdxStart: Int
            var gaussianIdxEnd: Int
            var cols: Int
            var rows: Int
            var pbShift: Float
            var maxValue: Float = 0
            var maxIdx: vDSP_Length = 0
            var mulLength: vDSP_Length
            var ret: [(InnerNote, PitchBend?)] = []
            for note in noteEvents {
                freqIdx = Int(midiPitchToContourBin(note.2).rounded())
                freqStartIdx = max(freqIdx - nBinsTolerance, 0)
                freqEndIdx = min(Constants.nFreqBinsContours, freqIdx + nBinsTolerance + 1)

                rows = note.1 - note.0
                cols = freqEndIdx - freqStartIdx
                var pitchBendSubMatrix = vDSP.ramp(withInitialValue: Float(0), increment: Float(0), count: cols)

                // gaussian 向量
                gaussianIdxStart = max(nBinsTolerance - freqIdx, 0)
                gaussianIdxEnd = windowLen - max(freqIdx - (Constants.nFreqBinsContours - nBinsTolerance - 1), 0)
                assert(gaussianIdxStart < freqGaussian.count && gaussianIdxEnd <= freqGaussian.count)
                let gaussianStart = pgaussian.baseAddress!.advanced(by: gaussianIdxStart)

                // 将子矩阵拆成行向量，和 gaussian 进行星乘运算
                var bends: PitchBend = []
                bends.reserveCapacity(rows)
                pbShift = -Float(nBinsTolerance - max(0, nBinsTolerance - freqIdx))
                for i in 0..<rows {
                    let start = (note.0 + i) * contourCols + freqStartIdx
                    let pstart = pcontours.baseAddress!.advanced(by: start)
                    mulLength = vDSP_Length(min(cols, gaussianIdxEnd - gaussianIdxStart))
                    vDSP_vmul(pstart, 1, gaussianStart, 1, &pitchBendSubMatrix, 1, mulLength)
                    // 求1个 bend
                    vDSP_maxvi(pitchBendSubMatrix, 1, &maxValue, &maxIdx, mulLength)
                    bends.append(Float(maxIdx))
                }
                vDSP.add(pbShift, bends, result: &bends)
                ret.append((note, bends.count > 0 ? bends : nil))
            }
            return ret
        }
    }
}

public actor NotesConverter {
    public struct Opt: Equatable, Sendable {
        public var onsetThreshold: Float  // 分割 - 合并 音符的力度，取值范围 [0.05, 0.95]
        public var frameThreshold: Float  // 更多 - 更少 由模型推理生成音符的置信度，取值范围 [0.05, 0.95]
        public var minNoteLength: Int  // 最短音符时长 [3, 50] ms
        public var minFreq: Float?  // 音调下限，单位 Hz [0, 2000]
        public var maxFreq: Float?  // 音调上限，单位 Hz [40, 3000]
        public var inferOnsets: Bool
        public var includePitchBends: Bool  // 弯音检测
        public var melodiaTrick: Bool  // 泛音检测
        public var energyThreshold: Int  // 能量限制

        public init(
            onsetThreshold: Float = 0.5,
            frameThreshold: Float = 0.3,
            minNoteLength: Int = 11,
            minFreq: Float? = nil,
            maxFreq: Float? = nil,
            inferOnsets: Bool = true,
            includePitchBends: Bool = true,
            melodiaTrick: Bool = true,
            energyThreshold: Int = 11
        ) {
            self.onsetThreshold = onsetThreshold
            self.frameThreshold = frameThreshold
            self.minNoteLength = minNoteLength
            self.minFreq = minFreq
            self.maxFreq = maxFreq
            self.inferOnsets = inferOnsets
            self.includePitchBends = includePitchBends
            self.melodiaTrick = melodiaTrick
            self.energyThreshold = energyThreshold
        }
    }

    public let onsets: MLMultiArray
    public let notes: MLMultiArray
    public let contours: MLMultiArray

    public init(onsets: MLMultiArray, notes: MLMultiArray, contours: MLMultiArray) {
        self.onsets = onsets
        self.notes = notes
        self.contours = contours
    }

    public func convert(_ opt: Opt = Opt()) throws -> [Note] {
        let estimatedNotes = outputToNotesPolyphonic(
            frames: notes,
            onsets: onsets,
            onsetThresh: opt.onsetThreshold,
            frameThresh: opt.frameThreshold,
            minNoteLen: opt.minNoteLength,
            inferOnsets: opt.inferOnsets,
            maxFreq: opt.maxFreq,
            minFreq: opt.minFreq,
            melodiaTrick: opt.melodiaTrick,
            energyTol: opt.energyThreshold
        )

        var estimatedNotesWithPitchBend: [(InnerNote, PitchBend?)]
        if opt.includePitchBends {
            estimatedNotesWithPitchBend = getPitchBends(contours: contours, noteEvents: estimatedNotes)
        } else {
            estimatedNotesWithPitchBend = estimatedNotes.map({ v in
                return (v, nil)
            })
        }

        let times = modelFrameToTime(contours.shape[0].intValue)
        return estimatedNotesWithPitchBend.map { v in
            return Note(
                startTime: times[v.0.0], endTime: times[v.0.1], pitch: v.0.2, amplitude: v.0.3, pitchBend: v.1)
        }
    }
}
