//
//  Constants.swift
//  test
//
//  Created by 张宇飞 on 2024/12/15.
//

struct Constants {
    static let fftHop = 256
    static let nOverlappingFrames = 30
    static let overlapLen = nOverlappingFrames * fftHop
    static let audioSampleRate = 22050
    static let audioWindowLen = 2
    // 模型输入的采样数量
    static let audioNSamples = audioSampleRate * audioWindowLen - fftHop
    static let hopSize = audioNSamples - overlapLen
    static let annotationsFps = audioSampleRate / fftHop
    static let midiOffset = 21
    static let maxFreqIdx = 87
    static let annotNFrames = annotationsFps * audioWindowLen
    static let contoursBinsPerSemitone = 3
    static let annotationsNSemitones = 88
    static let annotationsBaseFrequency = Float(27.5)
    static let nFreqBinsContours = annotationsNSemitones * contoursBinsPerSemitone
    static let nPitchBendTicks = 8192
}
