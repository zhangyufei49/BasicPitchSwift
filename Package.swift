// swift-tools-version: 5.4
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "BasicPitchSwift",
    platforms: [.macOS("13.5"), .iOS("16")],
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "BasicPitch",
            targets: ["BasicPitch"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/orchetect/MIDIKit.git", .upToNextMajor(from: "0.9.9")),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .target(
            name: "BasicPitch",
            dependencies: [.product(name: "MIDIKitSMF", package: "MIDIKit")],
            resources: [.process("nmp.mlmodelc")]
        ),
    ]
)
