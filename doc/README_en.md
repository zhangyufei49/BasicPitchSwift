## BasicPitchSwift

[BasicPitch](https://basicpitch.spotify.com/) is a audio-to-MIDI converter, built by Spotify.

It is developed by Python, and this project is it's Swift port.

## Example code

The minimum supported system version is 13.5 on MacOS

```swift
import BasicPitch

private func processAudio(_ audioFile: URL) async -> NoteCreation {
    // predict in worker thread, and monitor the progress
    return await Task.detached {
        return try! await BasicPitch.predict(audioFile) { name, value in
            // NOTE: not in the main thread
            print(name, value)
        }
    }.value
}

// save midi file
Task {
    let noteCreation = await processAudio(audioFileForReading)

    // genMidiFileData(_ opt:) has many options, this is the default behavior
    if let data = try? noteCreation?.genMidiFileData() {
        try? data.write(to: midiFileForWriting)
    }
}
```

## Technical fundamentals

**BasicPitch** working with 3 steps:

1. predict audio frames with **nmp** model
2. Use the **Post-Processing algorithm** to process the output of the **nmp** model
3. Convert the data from the previous step to a MIDI file

This project also follows this steps:

1. Use the CoreML model version of nmp for inference
2. Use Apple's **Accelerate** framework to port the post-processing algorithms implemented by **Numpy** and **Scipy** in the Python version
3. Use **MIDIKit** to generate a MIDI file

The audio data entered into the nmp model in step 1 will be a little different from the Python version:

- The Python version uses a mean value of multi-channel data as the downmix algorithm, and the resampling algorithm is **soxr_hq**
- This project uses Apple's own **AVAudioConverter** to achieve these two steps. The output result cannot be strictly consistent with the Python version, but the final result is basically the same.

## PS

This project is less annotated and is basically a one-to-one port of the Python version.

If there is anything that seems confusing, you can look at the Python version of the comment by the function name.
