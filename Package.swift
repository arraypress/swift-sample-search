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

// Swift 7's import rule, today: a file must import every module whose members
// it uses, rather than leaning on another file's imports.
for target in package.targets where target.type != .system && target.type != .binary && target.type != .plugin {
    target.swiftSettings = (target.swiftSettings ?? []) + [.enableUpcomingFeature("MemberImportVisibility")]
}
