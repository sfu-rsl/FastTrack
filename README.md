# FastTrack

FastTrack is an optimized visual-inertial tracking module designed to harness GPU computing power, significantly speeding up the most computationally intensive aspects of SLAM (Simultaneous Localization and Mapping). Our implementation is based on the well-established ORB-SLAM3 system.

## Key Features
- GPU Acceleration: We've developed several GPU kernels to offload critical components of the tracking process, enhancing overall performance.
- Stereo Feature Matching: Specialized kernels accelerate stereo feature matching for both pinhole and fisheye cameras.
- Map Point Projection: Our implementation includes efficient GPU kernels for searching map points by projection, improving the discovery of visible map points in real-time.
- Optimized ORB Feature Extraction: We integrate existing acceleration techniques for ORB feature extraction.
- Efficient Data Handling: Our design includes optimized data transfer processes between tracking components and bypasses pose optimization to enhance speed.
