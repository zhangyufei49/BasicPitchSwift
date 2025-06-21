//
//  Note.swift
//  BasicPitchSwift
//
//  Created by 张宇飞 on 2025/6/18.
//

import Foundation

public typealias PitchBend = [Float]

public struct Note: Sendable {
    let startTime: Float
    let endTime: Float
    let pitch: Int
    let amplitude: Float
    var pitchBend: PitchBend?
}

extension Note: Comparable {
    public static func < (lhs: Note, rhs: Note) -> Bool {
        if lhs.startTime != rhs.startTime {
            return lhs.startTime < rhs.startTime
        }

        if lhs.endTime != rhs.endTime {
            return lhs.endTime < rhs.endTime
        }

        if lhs.pitch != rhs.pitch {
            return lhs.pitch < rhs.pitch
        }

        if lhs.amplitude != rhs.amplitude {
            return lhs.amplitude < rhs.amplitude
        }

        let l = (lhs.pitchBend != nil) ? lhs.pitchBend!.count : -1

        let r = (rhs.pitchBend != nil) ? rhs.pitchBend!.count : -1
        return l <= r
    }
}
