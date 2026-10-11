# Copyright (C) 2026 Teng-Shuo Zhang
# SPDX-License-Identifier: Apache-2.0

const NOCI_PROJECT_PATH = normpath(joinpath(@__DIR__, "..", "..", "Project.toml"))
const NOCI_COMPATIBILITY_PATH = normpath(joinpath(@__DIR__, "..", "..", "OpenMolcasCompatibility.toml"))
Base.include_dependency(NOCI_PROJECT_PATH)
Base.include_dependency(NOCI_COMPATIBILITY_PATH)
const NOCI_VERSION = VersionNumber(TOML.parsefile(NOCI_PROJECT_PATH)["version"])
const NOCI_BACKEND_COMPATIBILITY = TOML.parsefile(NOCI_COMPATIBILITY_PATH)

"""
    noci_version()

Return this NOCI package's full calendar release version as a `VersionNumber`.
The identity comes only from the package's own `Project.toml`, which participates in precompile invalidation; the active project and working directory are ignored.
No scientific input is mutated, no backend or GPU is initialized, and missing or invalid package metadata fails during module loading.
"""
noci_version() = NOCI_VERSION

"""
    noci_gpu_supported()

Return whether this release declares GPU support in its immutable product profile.
The Boolean has no physical units, no state is mutated, and no CUDA module or device is loaded.
"""
noci_gpu_supported() = Bool(NOCI_BACKEND_COMPATIBILITY["release"]["gpu"])
