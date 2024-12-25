[English](doc/README_en.md)

## BasicPitchSwift

[BasicPitch](https://basicpitch.spotify.com/) 是 Spotify 的一个音频转 MIDI 的项目。</br>
它本身由 Python 开发，本项目为其 Swift 移植版本。

## 使用方法

MacOS 为例,最低支持系统版本为 13.5

```swift
import BasicPitch

private func processAudio(_ audioFile: URL) async -> NoteCreation {
    // 在一个后台线程中进行预测，并获取操作的进度
    return await Task.detached {
        return try! await BasicPitch.predict(audioFile) { name, value in
            // NOTE: 注意，这里非主线程
            print(name, value)
        }
    }.value
}

// 保存 MIDI 文件
Task {
    let noteCreation = await processAudio(audioFileForReading)

    // genMidiFileData 有很多可配置的参数，这里是默认操作
    if let data = try? noteCreation?.genMidiFileData() {
        try? data.write(to: midiFileForWriting)
    }
}
```

## 技术基本原理

**BasicPitch** 本身由 3 个步骤来完成工作:

1. 使用 **nmp** 模型推理音频输入
2. 使用 **后处理** 算法处理 **nmp** 模型的输出
3. 将上一步的数据转换为 MIDI 文件

本项目也是遵循这个原理:

1. 使用 **nmp** 的 CoreML 模型版本进行推理
2. 使用苹果的 **Accelerate** 框架移植 Python 版中由 Numpy/scipy 实现的后处理算法
3. 使用 **MIDIKit** 生成 MIDI 文件

其中，第 1 步中输入到 **nmp** 模型的音频数据会和 Python 版本有一些不同:

- Python 版本使用到的 downmix 算法为求多声道平均，用到的重采样算法为 **soxr_hq**
- 本项目使用苹果自己的 **AVAudioConverter** 来实现这两步，因为算法黑盒，所以输出结果不能与 Python 版本严格一致，但是最终结果基本相同。

## 其它说明

本项目注释较少，基本上是对 Python 版本的一比一移植。</br>
有任何看起来迷惑的地方可以按照函数名去看 Python 版本的注释。
