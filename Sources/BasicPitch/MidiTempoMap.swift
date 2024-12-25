//
//  MidiTempoMap.swift
//  test
//
//  Created by 张宇飞 on 2024/12/15.
//

// 一个简化的 tempo map，不考虑 bpm 和 beatsUnit 的变化
// 只适用于 basic-pitch 转换 midi 的场景
struct MidiTempoMap {
    let bpm: UInt
    let tpq: UInt16
    let beatUnit: UInt16

    init(bpm: UInt = 120, tpq: UInt16 = 480, beatUnit: UInt16 = 4) {
        self.bpm = bpm
        self.tpq = tpq
        self.beatUnit = beatUnit
    }

    func secsToTicks(_ secs: Float) -> UInt32 {
        if secs <= 0 {
            return 0
        }

        let r = (secs * Float(tpq) * Float(bpm) / (Float(15.0) * Float(beatUnit))).rounded()
        return UInt32(r)
    }
}
