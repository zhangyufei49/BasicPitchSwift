//
//  MidiWriter.swift
//  BasicPitchSwift
//
//  Created by 张宇飞 on 2025/6/18.
//

import Accelerate
import Foundation
import MIDIKitSMF

private func dropOpverlappingPitchBends(_ notes: [Note]) -> [Note] {
    if notes.isEmpty {
        return notes
    }
    var ret = notes.sorted()
    for i in 0..<(ret.count - 1) {
        for j in (i + 1)..<ret.count {
            if ret[j].startTime >= ret[i].endTime {
                break
            }
            ret[i].pitchBend = nil
            ret[j].pitchBend = nil
        }
    }
    return ret
}

private func getMidiEventScore(_ event: MIDIEvent) -> Int {
    switch event {
    case .noteOn:
        let v = event.midi1RawDataBytes()!
        return Int(v.0!) * 1000 + Int(v.1!)
    case .pitchBend:
        return 2
    case .programChange:
        return 1
    default:
        return 0
    }
}

public actor MidiWriter {
    public struct Opt: Equatable, Sendable {
        public var midiTempo: Int  // midi 文件使用的默认拍速 [24, 224]
        public var midiProgram: Int  // midi 文件使用的默认乐器号
        public var multiplePitchBends: Bool  // 每个 pitch 一个单独的音轨

        public init(
            midiTempo: Int = 120,
            midiProgram: Int = 4,
            multiplePitchBends: Bool = false
        ) {
            self.midiTempo = midiTempo
            self.midiProgram = midiProgram
            self.multiplePitchBends = multiplePitchBends
        }
    }

    public let notes: [Note]

    public init(notes: [Note]) {
        self.notes = notes;
    }

    public func write(_ opt: Opt = Opt()) -> MIDIFile {
        let notes = opt.multiplePitchBends ? self.notes : dropOpverlappingPitchBends(self.notes)

        let tm = MidiTempoMap(bpm: UInt(opt.midiTempo))
        // pitch: (ticks, events)
        var tracks: [Int: [(UInt32, MIDIEvent)]] = [:]
        var orderedTrackIdxs: [Int] = []
        let defaultProgram = UInt7(opt.midiProgram)
        var channelCounter = -1
        var trackIdx: Int
        var channel: UInt4
        var vel: UInt7
        let fNPitchBendTicks = Float(Constants.nPitchBendTicks)
        let pitchBendTicksScalar = Float(4096) / Float(Constants.contoursBinsPerSemitone)
        let bendRange = (Float(0)...Float(Constants.nPitchBendTicks * 2 - 1))
        for n in notes {
            trackIdx = opt.multiplePitchBends ? n.pitch : 0

            if tracks[trackIdx] == nil {
                channelCounter += 1
                channel = UInt4(channelCounter % 16)
                orderedTrackIdxs.append(trackIdx)
                tracks[trackIdx] = [
                    (0, .programChange(program: defaultProgram, channel: channel))
                ]
            } else {
                channel = tracks[trackIdx]!.first!.1.channel!
            }

            // note on/ note off
            vel = UInt7((127 * n.amplitude).rounded())
            tracks[trackIdx]!.append(
                (
                    tm.secsToTicks(n.startTime),
                    .noteOn(.init(note: UInt7(n.pitch), velocity: .midi1(vel), channel: channel))
                )
            )

            // pitch bend
            if n.pitchBend != nil {
                let pitchBendTimes = vDSP.ramp(in: n.startTime...n.endTime, count: n.pitchBend!.count)
                var fticks = vDSP.multiply(pitchBendTicksScalar, n.pitchBend!)
                // 这里不同于 pretty midi [-8192, 8191] 的 bend pitch 范围
                // MIDIKit 可用的是GM 中的设定的 [0, 0x3fff] 所以，进行了一个转换
                vForce.nearestInteger(fticks, result: &fticks)
                vDSP.add(fNPitchBendTicks, fticks, result: &fticks)
                vDSP.clip(fticks, to: bendRange, result: &fticks)
                let pitchBendMidiTicks =
                    vDSP.floatingPointToInteger(fticks, integerType: Int32.self, rounding: .towardNearestInteger)
                for (pbTime, pbTick) in zip(pitchBendTimes, pitchBendMidiTicks) {
                    tracks[trackIdx]!.append(
                        (
                            tm.secsToTicks(pbTime),
                            .pitchBend(.init(value: .midi1(UInt14(Int(pbTick))), channel: channel))
                        ))
                }
            }

            tracks[trackIdx]!.append(
                (
                    tm.secsToTicks(n.endTime),
                    .noteOn(.init(note: UInt7(n.pitch), velocity: .midi1(0), channel: channel))
                )
            )
        }

        var mid = MIDIFile(
            format: .multipleTracksSynchronous,
            timeBase: .musical(ticksPerQuarterNote: tm.tpq),
            chunks: [
                .track([
                    .tempo(bpm: Double(tm.bpm)),
                    .timeSignature(numerator: 4, denominator: 2),
                ])
            ]
        )

        for i in orderedTrackIdxs {
            // 将 tracks 中的数据按照 ticks 排序
            tracks[i]!.sort { l, r in
                if l.0 < r.0 {
                    return true
                }
                if l.0 > r.0 {
                    return false
                }
                // 相等的情况下，需要判定类型
                return getMidiEventScore(l.1) <= getMidiEventScore(r.1)
            }
            // 生成 MIDIFileEvent
            var lastTicks: UInt32 = 0
            var events: [MIDIFileEvent] = []
            for i in tracks[i]! {
                lastTicks = i.0 - lastTicks
                events.append(i.1.smfEvent(delta: .ticks(lastTicks))!)
                lastTicks = i.0
            }
            mid.chunks.append(.track(events))
        }

        return mid
    }

}
