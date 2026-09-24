// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "swift-sample-search",
    platforms: [
        .macOS("27.0"),
    ],
    products: [
        .library(name: "SampleSearch", targets: ["SampleSearch"]),
    ],
    targets: [
        .target(name: "SampleSearch"),
        .testTarget(
            name: "SampleSearchTests",
            dependencies: ["SampleSearch"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
