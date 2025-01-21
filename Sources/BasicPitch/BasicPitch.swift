//
//  BasicPitch.swift
//  ap
//
//  Created by 张宇飞 on 2024/11/16.
//

import AVFAudio
import Accelerate
import CoreML
import Foundation

public enum BasicPitchError: Error {
    case invalidAudioFormat
    case noAudioData
}

public actor BasicPitch {
    /// 预热模型。这样在使用的时候加载会更快
    public static func warmup() {
        if let data = try? MLMultiArray(shape: [1, NSNumber(value: Constants.audioNSamples), 1], dataType: .float32) {
            _ = try? nmp(configuration: .init()).prediction(input_2: data)
        }
    }
}

extension BasicPitch {
    public enum ProgressName: Sendable {
        case readingAudio
        case resampleAudio
        case prediction
    }

    public typealias OnProgressChanged = @Sendable (ProgressName, Double) -> Void

    /// 用于预测的音频编码格式
    public static var targetAudioFormat: AVAudioFormat {
        return AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(
                Constants.audioSampleRate
            ),
            channels: 1,
            interleaved: false
        )!
    }

    public static func predict(_ audioFile: URL, _ onProgressChanged: OnProgressChanged? = nil) throws -> NoteCreation {
        let f = try! AVAudioFile(forReading: audioFile)
        var audio: AVAudioPCMBuffer
        if f.processingFormat == BasicPitch.targetAudioFormat {
            // 如果格式符合预期，直接全部读取
            audio = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length))!
            onProgressChanged?(.readingAudio, 0.0)
            try! f.read(into: audio, frameCount: audio.frameCapacity)
            onProgressChanged?(.readingAudio, 1.0)
            onProgressChanged?(.resampleAudio, 0.0)
            onProgressChanged?(.resampleAudio, 1.0)
        } else {
            // 读取文件并进行重采样
            audio = resample(try! loadAudio(f, onProgressChanged), onProgressChanged)
        }

        // 进行预测
        return try! BasicPitch.predict(audio, onProgressChanged)
    }

    public static func predict(_ audio: AVAudioPCMBuffer, _ onProgressChanged: OnProgressChanged? = nil) throws
        -> NoteCreation
    {
        if audio.format != targetAudioFormat {
            throw BasicPitchError.invalidAudioFormat
        }
        if audio.frameLength == 0 {
            throw BasicPitchError.noAudioData
        }

        let model = try! nmp(configuration: .init())
        let audioFrames = UnsafeBufferPointer(start: audio.floatChannelData![0], count: Int(audio.frameLength))
        var modelOutput = ModelOutput()
        var it = ModelInputIterator(audioFrames)
        while let i = it.next() {
            // 进行一次模型预测
            if let out = try? model.prediction(input: i.0) {
                modelOutput.contours.append(out.Identity)
                modelOutput.notes.append(out.Identity_1)
                modelOutput.onsets.append(out.Identity_2)
            }
            onProgressChanged?(ProgressName.prediction, i.1 * 0.99)
        }

        // 转换模型预测结果数组为单一的 MLMultiArray 对象，用于生成音符
        let (notes, onsets, contours) = modelOutput.unwrapped(Float(audio.frameLength))
        onProgressChanged?(ProgressName.prediction, 1.0)

        return NoteCreation(onsets: onsets, notes: notes, contours: contours)
    }

}

private struct ModelInputIterator: IteratorProtocol {
    // (model input, progress)
    typealias Element = (nmpInput, Double)

    private let modelInput: nmpInput
    private let audioFrames: UnsafeBufferPointer<Float>
    private var i = Constants.overlapLen / -2

    init(_ audioFrames: UnsafeBufferPointer<Float>) {
        self.audioFrames = audioFrames
        self.modelInput = nmpInput(
            input_2: try! MLMultiArray(shape: [1, NSNumber(value: Constants.audioNSamples), 1], dataType: .float32)
        )
    }

    mutating public func next() -> Element? {
        // 停止条件
        let totalFrames = audioFrames.count
        if self.i >= totalFrames {
            return nil
        }

        // 填充模型输入数据
        let limit = Constants.audioNSamples
        let input = self.modelInput.input_2

        input.withUnsafeMutableBufferPointer(ofType: Float.self) { dst, _ in
            // 当 i < 0 时，需要补0
            var offset = 0
            if self.i < 0 {
                offset = -self.i
                memset(dst.baseAddress!, 0, offset * MemoryLayout<Float>.size)
            }
            // 复制音频采样数据
            let j = max(0, self.i)
            let n = min(limit - offset, totalFrames - j)
            memcpy(
                dst.baseAddress!.advanced(by: offset),
                audioFrames.baseAddress!.advanced(by: j),
                n * MemoryLayout<Float>.size
            )
            // 对于最后一次预测的输入数据不足的情况进行补 0
            offset += n
            if offset < limit {
                memset(dst.baseAddress!.advanced(by: offset), 0, (limit - offset) * MemoryLayout<Float>.size)
            }
        }

        // 每次指针前进 hopSize
        self.i += Constants.hopSize

        // 计算进度
        let halfOverLapLen = Constants.overlapLen / 2
        let total = Double(totalFrames + halfOverLapLen)
        let now = Double(self.i + halfOverLapLen)
        return (modelInput, min(now / total, 1.0))
    }
}

private struct ModelOutput {
    var notes: [MLMultiArray] = []
    var onsets: [MLMultiArray] = []
    var contours: [MLMultiArray] = []

    mutating func unwrapped(_ totalFrames: Float) -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        return (
            ModelOutput.unwrapped(&self.notes, totalFrames: totalFrames),
            ModelOutput.unwrapped(&self.onsets, totalFrames: totalFrames),
            ModelOutput.unwrapped(&self.contours, totalFrames: totalFrames)
        )
    }

    private static func unwrapped(_ arr: inout [MLMultiArray], totalFrames: Float) -> MLMultiArray {
        let nOlap = Constants.nOverlappingFrames / 2
        let nOutputFramesOri = Int(totalFrames * Float(Constants.annotationsFps) / Float(Constants.audioSampleRate))
        let oriShape = [arr.count, arr[0].shape[1].intValue]
        let shape0 = min(oriShape[0] * oriShape[1] - nOlap * 2, nOutputFramesOri)
        let jLimit = oriShape[1] - nOlap
        let ret = try! MLMultiArray(
            shape: [NSNumber(value: shape0), arr[0].shape[2]], dataType: arr[0].dataType)
        var size: Int = 0
        ret.withUnsafeMutableBytes({ dst, strides in
            for i in 0..<arr.count {
                let srcStep = arr[i].strides[1].intValue
                arr[i].withUnsafeBytes({ src in
                    for j in nOlap..<jLimit {
                        let dp = dst.baseAddress!.advanced(by: size * MemoryLayout<Float>.size * strides[0])
                        let sp = src.baseAddress!.advanced(by: j * MemoryLayout<Float>.size * srcStep)
                        memcpy(dp, sp, MemoryLayout<Float>.size * arr[0].shape[2].intValue)
                        size += 1
                        if size == shape0 {
                            break
                        }
                    }
                })
                if size == shape0 {
                    break
                }
            }
        })
        // 释放内存
        arr.removeAll()
        return ret
    }
}

/// 将输入的音频数据转换为 BasicPitch.targetAudioFormat
private func resample(_ buf: AVAudioPCMBuffer, _ onProgressChanged: BasicPitch.OnProgressChanged? = nil)
    -> AVAudioPCMBuffer
{
    onProgressChanged?(.resampleAudio, 0.0)
    let targetFormat = BasicPitch.targetAudioFormat
    let converter = AVAudioConverter(from: buf.format, to: targetFormat)!
    converter.downmix = true
    converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering

    onProgressChanged?(.resampleAudio, 0.1)
    let inputCallback: AVAudioConverterInputBlock = { inNumPackets, outStatus in
        outStatus.pointee = AVAudioConverterInputStatus.haveData
        return buf
    }
    let n = (Double(buf.frameLength) * targetFormat.sampleRate / Double(buf.format.sampleRate)).rounded()
    let result = AVAudioPCMBuffer(
        pcmFormat: targetFormat,
        frameCapacity: AVAudioFrameCount(n))!

    onProgressChanged?(.resampleAudio, 0.2)
    var error: NSError? = nil
    let status = converter.convert(to: result, error: &error, withInputFrom: inputCallback)
    assert(status != .error)

    onProgressChanged?(.resampleAudio, 1.0)
    return result
}

private func loadAudio(_ f: AVAudioFile, _ onProgressChanged: BasicPitch.OnProgressChanged? = nil) throws
    -> AVAudioPCMBuffer
{
    onProgressChanged?(.readingAudio, 0.0)
    let fmt = f.processingFormat
    let targetFmt = AVAudioFormat(
        commonFormat: fmt.commonFormat,
        sampleRate: fmt.sampleRate,
        channels: 1,
        interleaved: fmt.isInterleaved
    )!
    let ret = AVAudioPCMBuffer(pcmFormat: targetFmt, frameCapacity: AVAudioFrameCount(f.length))!
    ret.frameLength = ret.frameCapacity
    let rbuf = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(4096))!
    let wbuf = AVAudioPCMBuffer(pcmFormat: targetFmt, frameCapacity: rbuf.frameCapacity)!
    var readed: AVAudioFramePosition = 0
    let converter = AVAudioConverter(from: fmt, to: targetFmt)!
    converter.downmix = true

    // 分块读取音频文件的数据，然后进行 downmix 操作，这样可以降低内存占用
    onProgressChanged?(.readingAudio, 0.1)
    while readed < f.length {
        try! f.read(into: rbuf, frameCount: rbuf.frameCapacity)
        // downmix
        wbuf.frameLength = 0
        try! converter.convert(to: wbuf, from: rbuf)
        // 拷贝数据到 ret
        switch fmt.commonFormat {
        case .pcmFormatFloat32:
            memcpy(
                ret.floatChannelData![0].advanced(by: Int(readed)),
                wbuf.floatChannelData![0],
                Int(wbuf.frameLength) * MemoryLayout<Float32>.size
            )
        case .pcmFormatFloat64:
            memcpy(
                ret.floatChannelData![0].advanced(by: Int(readed)),
                wbuf.floatChannelData![0],
                Int(wbuf.frameLength) * MemoryLayout<Float64>.size
            )
        case .pcmFormatInt16:
            memcpy(
                ret.int16ChannelData![0].advanced(by: Int(readed)),
                wbuf.int16ChannelData![0],
                Int(wbuf.frameLength) * MemoryLayout<Int16>.size
            )
        case .pcmFormatInt32:
            memcpy(
                ret.int32ChannelData![0].advanced(by: Int(readed)),
                wbuf.int32ChannelData![0],
                Int(wbuf.frameLength) * MemoryLayout<Int32>.size
            )
        default:
            break
        }

        // 调整进度
        readed += AVAudioFramePosition(rbuf.frameLength)
        onProgressChanged?(.readingAudio, 0.1 + 0.9 * Double(readed) / Double(ret.frameCapacity))
    }

    return ret
}
