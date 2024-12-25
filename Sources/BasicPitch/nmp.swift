//
// nmp.swift
//
// This file was automatically generated and should not be edited.
//

import CoreML


/// Model Prediction Input Type
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class nmpInput : MLFeatureProvider {

    /// input_2 as 1 × 43844 × 1 3-dimensional array of floats
    var input_2: MLMultiArray

    var featureNames: Set<String> { ["input_2"] }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        if featureName == "input_2" {
            return MLFeatureValue(multiArray: input_2)
        }
        return nil
    }

    init(input_2: MLMultiArray) {
        self.input_2 = input_2
    }

    convenience init(input_2: MLShapedArray<Float>) {
        self.init(input_2: MLMultiArray(input_2))
    }

}


/// Model Prediction Output Type
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class nmpOutput : MLFeatureProvider {

    /// Source provided by CoreML
    private let provider : MLFeatureProvider

    /// Identity as 1 × 172 × 264 3-dimensional array of floats
    var Identity: MLMultiArray {
        provider.featureValue(for: "Identity")!.multiArrayValue!
    }

    /// Identity as 1 × 172 × 264 3-dimensional array of floats
    var IdentityShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(Identity)
    }

    /// Identity_1 as 1 × 172 × 88 3-dimensional array of floats
    var Identity_1: MLMultiArray {
        provider.featureValue(for: "Identity_1")!.multiArrayValue!
    }

    /// Identity_1 as 1 × 172 × 88 3-dimensional array of floats
    var Identity_1ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(Identity_1)
    }

    /// Identity_2 as 1 × 172 × 88 3-dimensional array of floats
    var Identity_2: MLMultiArray {
        provider.featureValue(for: "Identity_2")!.multiArrayValue!
    }

    /// Identity_2 as 1 × 172 × 88 3-dimensional array of floats
    var Identity_2ShapedArray: MLShapedArray<Float> {
        MLShapedArray<Float>(Identity_2)
    }

    var featureNames: Set<String> {
        provider.featureNames
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        provider.featureValue(for: featureName)
    }

    init(Identity: MLMultiArray, Identity_1: MLMultiArray, Identity_2: MLMultiArray) {
        self.provider = try! MLDictionaryFeatureProvider(dictionary: ["Identity" : MLFeatureValue(multiArray: Identity), "Identity_1" : MLFeatureValue(multiArray: Identity_1), "Identity_2" : MLFeatureValue(multiArray: Identity_2)])
    }

    init(features: MLFeatureProvider) {
        self.provider = features
    }
}

/// Class for model loading and prediction
@available(macOS 12.0, iOS 15.0, tvOS 15.0, watchOS 8.0, visionOS 1.0, *)
class nmp {
    let model: MLModel

    /// URL of model assuming it was installed in the same bundle as this class
    class var urlOfModelInThisBundle : URL {
        return Bundle.module.url(forResource: "nmp", withExtension: "mlmodelc", subdirectory: nil)!
    }

    /**
        Construct nmp instance with an existing MLModel object.

        Usually the application does not use this initializer unless it makes a subclass of nmp.
        Such application may want to use `MLModel(contentsOfURL:configuration:)` and `nmp.urlOfModelInThisBundle` to create a MLModel object to pass-in.

        - parameters:
          - model: MLModel object
    */
    init(model: MLModel) {
        self.model = model
    }

    /**
        Construct a model with configuration

        - parameters:
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(configuration: MLModelConfiguration = MLModelConfiguration()) throws {
        try self.init(contentsOf: type(of:self).urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct nmp instance with explicit path to mlmodelc file
        - parameters:
           - modelURL: the file url of the model

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL) throws {
        try self.init(model: MLModel(contentsOf: modelURL))
    }

    /**
        Construct a model with URL of the .mlmodelc directory and configuration

        - parameters:
           - modelURL: the file url of the model
           - configuration: the desired model configuration

        - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf modelURL: URL, configuration: MLModelConfiguration) throws {
        try self.init(model: MLModel(contentsOf: modelURL, configuration: configuration))
    }

    /**
        Construct nmp instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<nmp, Error>) -> Void) {
        load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration, completionHandler: handler)
    }

    /**
        Construct nmp instance asynchronously with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - configuration: the desired model configuration
    */
    class func load(configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> nmp {
        try await load(contentsOf: self.urlOfModelInThisBundle, configuration: configuration)
    }

    /**
        Construct nmp instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
          - handler: the completion handler to be called when the model loading completes successfully or unsuccessfully
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration(), completionHandler handler: @escaping (Swift.Result<nmp, Error>) -> Void) {
        MLModel.load(contentsOf: modelURL, configuration: configuration) { result in
            switch result {
            case .failure(let error):
                handler(.failure(error))
            case .success(let model):
                handler(.success(nmp(model: model)))
            }
        }
    }

    /**
        Construct nmp instance asynchronously with URL of the .mlmodelc directory with optional configuration.

        Model loading may take time when the model content is not immediately available (e.g. encrypted model). Use this factory method especially when the caller is on the main thread.

        - parameters:
          - modelURL: the URL to the model
          - configuration: the desired model configuration
    */
    class func load(contentsOf modelURL: URL, configuration: MLModelConfiguration = MLModelConfiguration()) async throws -> nmp {
        let model = try await MLModel.load(contentsOf: modelURL, configuration: configuration)
        return nmp(model: model)
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as nmpInput

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as nmpOutput
    */
    func prediction(input: nmpInput) throws -> nmpOutput {
        try prediction(input: input, options: MLPredictionOptions())
    }

    /**
        Make a prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as nmpInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as nmpOutput
    */
    func prediction(input: nmpInput, options: MLPredictionOptions) throws -> nmpOutput {
        let outFeatures = try model.prediction(from: input, options: options)
        return nmpOutput(features: outFeatures)
    }

    /**
        Make an asynchronous prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - input: the input to the prediction as nmpInput
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as nmpOutput
    */
    @available(macOS 14.0, iOS 17.0, tvOS 17.0, watchOS 10.0, visionOS 1.0, *)
    func prediction(input: nmpInput, options: MLPredictionOptions = MLPredictionOptions()) async throws -> nmpOutput {
        let outFeatures = try await model.prediction(from: input, options: options)
        return nmpOutput(features: outFeatures)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - input_2: 1 × 43844 × 1 3-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as nmpOutput
    */
    func prediction(input_2: MLMultiArray) throws -> nmpOutput {
        let input_ = nmpInput(input_2: input_2)
        return try prediction(input: input_)
    }

    /**
        Make a prediction using the convenience interface

        It uses the default function if the model has multiple functions.

        - parameters:
            - input_2: 1 × 43844 × 1 3-dimensional array of floats

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as nmpOutput
    */

    func prediction(input_2: MLShapedArray<Float>) throws -> nmpOutput {
        let input_ = nmpInput(input_2: input_2)
        return try prediction(input: input_)
    }

    /**
        Make a batch prediction using the structured interface

        It uses the default function if the model has multiple functions.

        - parameters:
           - inputs: the inputs to the prediction as [nmpInput]
           - options: prediction options

        - throws: an NSError object that describes the problem

        - returns: the result of the prediction as [nmpOutput]
    */
    func predictions(inputs: [nmpInput], options: MLPredictionOptions = MLPredictionOptions()) throws -> [nmpOutput] {
        let batchIn = MLArrayBatchProvider(array: inputs)
        let batchOut = try model.predictions(from: batchIn, options: options)
        var results : [nmpOutput] = []
        results.reserveCapacity(inputs.count)
        for i in 0..<batchOut.count {
            let outProvider = batchOut.features(at: i)
            let result =  nmpOutput(features: outProvider)
            results.append(result)
        }
        return results
    }
}
