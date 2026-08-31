### **Revised roadmap — built for execution, not just understanding**

#### **Phase 0: Environment & Sanity Check (this week, before any "real" work)**

Before touching ImageCAS, you need to prove your machine/Colab/cluster can actually do 3D training at all.

1. Set up a Python environment with `torch`, `monai`, `nibabel`, `itk`, `matplotlib`. Use a CLI tool (Claude Code, terminal) to script this as a `requirements.txt` \+ virtual env, not manual pip installs across 5 laptops.  
2. Confirm GPU access and VRAM (`nvidia-smi`). This number determines your patch size later — write it down now.  
3. Download **one single ImageCAS case** (not the full \~1000). Load it with `nibabel`, print its shape, visualize one slice. This is your "hello world" — if this works, your stack works.  
4. Decide and document team compute: whose GPU, Colab Pro, or university cluster. This single decision changes your patch size, batch size, and how long Phase 2 will take, so lock it in now rather than mid-pipeline.

#### **Phase 1: ImageCAS Pipeline** 

**Data Preparation:** Download the ImageCAS dataset. Ensure the files are in standard medical formats like NIfTI (`.nii.gz`).

**Implement 3D Transforms:** Use MONAI's dictionary transforms to load the images, normalize the voxel spacing (spacing the data consistently so the heart size is uniform), and scale the intensities.

**Build the Tiling Dataloader:** Implement MONAI's `RandCropByPosNegLabeld`. This transform acts as your "tiler." It extracts random 3D patches from the CT scan, ensuring it grabs patches that actually contain the coronary artery (positive patches) rather than just empty lung tissue (negative patches).

**The addition (claude addition):** explicitly inspect ImageCAS's label format before writing code. Open 2-3 ground truth masks and confirm: is it a full artery centerline/tree segmentation, or a stenosis-region mask? This single fact determines whether your hospital annotation task should mirror it. Do this *now*, write it down, and tell your radiologist exactly what label format you need before they start, not after.

#### **Phase 2: Architecture & Training** 

**Define the Model:** Your poster specifies a Residual U-Net with Group Normalization (GN). Instead of writing complex 3D convolution blocks from scratch, instantiate MONAI's `SegResNet`. It perfectly matches your poster's requirements (Residual blocks, GN, capable of Deep Supervision) and is proven for 3D medical segmentation.  
**Loss Function Setup:** Medical datasets are highly imbalanced (the blocked artery is tiny compared to the whole chest). Combine standard Cross-Entropy with Dice Loss to force the model to care about the small foreground structures.  
**Train & Checkpoint:** Train this model on the ImageCAS dataset. Save the weights. This model will now understand what a coronary artery looks like, even if it isn't perfectly tailored to local hospital machines yet.

You need to make an explicit choice here rather than defaulting silently:

* **Option A (recommended given your timeline):** Train SegResNet only, as Gemini suggested, and reframe your thesis contribution as "comparative evaluation \+ hybrid fusion design," even if the hybrid is only fully realized as future work in your conclusion. This is honest and defensible in defense.  
* **Option B:** Actually train all four (3D U-Net, V-Net, SegResNet, nnU-Net) as your poster shows, then build the fusion model. This is 3-4x the compute and time of Option A.

Given September/October and "zero idea of 3D Tiled CNN" as your starting point, I'd push you hard toward A unless your supervisor explicitly requires B. This is a conversation to have with Dr. Alam this week, not something to discover in August.

#### **Phase 3: Local Data Integration (the part most likely to slip)**

Gemini frames this as "by this time your radiologists should have annotated a portion of the dataset" — that's an assumption, not a plan. Make it a plan:

1. This week: reach out to multiple radiologists/hospitals in parallel, not sequentially. Annotation is almost always the long pole.  
2. Negotiate a **partial delivery schedule** — e.g., 20 annotated scans by mid-July as a first batch, rather than waiting for all 150-160 at once. This lets you start Phase 3 fine-tuning on a small set while annotation continues, which directly mirrors the parallel-track philosophy Gemini correctly recommended for ImageCAS vs. hospital data.  
3. Build a **fallback**: if by August 1st you have fewer than \~20-30 annotated hospital scans, your defense story becomes "trained and validated on ImageCAS, hospital data shown as qualitative case studies / preliminary fine-tuning results" rather than a full retrain. Decide this fallback now, calmly, rather than panicking in September.

The format conversion (DICOM → NIfTI) and transfer learning approach Gemini describes is correct.

#### **Phase 4: Evaluation & Defense Prep (matches Gemini's Phase 4\)**

DSC \> 0.85 and HD95 \< 2.0mm as targets, ITK-SNAP/3D Slicer for visualization — all correct and matches what your poster already claims. No changes.

 **New strategy to implement**

### **1\. Automatic Mixed Precision (AMP)**

If not already implemented, this is the single most impactful optimization remaining. By default, PyTorch trains using 32-bit floating-point numbers (`float32`). AMP automatically switches safe operations (like convolutions) to 16-bit floating-point (`float16`) while keeping critical operations (like loss calculations) in `float32` to maintain numerical stability.

* **Impact:** Cuts VRAM usage nearly in half, allowing for larger batch sizes or patch sizes, and accelerates mathematical operations on modern GPU Tensor Cores.  
* **Implementation:**

Python

from torch.cuda.amp import autocast, GradScaler

scaler \= GradScaler()

for batch\_data in train\_loader:

    inputs, labels \= batch\_data\["image"\].to(device), batch\_data\["label"\].to(device)

    optimizer.zero\_grad()

    

    \# Runs the forward pass in mixed precision

    with autocast():

        outputs \= model(inputs)

        loss \= loss\_function(outputs, labels)

    

    \# Scales the loss and calls backward

    scaler.scale(loss).backward()

    scaler.step(optimizer)

    scaler.update()

### **2\. Enable TensorFloat-32 (TF32) Execution**

Modern GPU architectures support TensorFloat-32 execution modes. This allows the GPU to use internal 19-bit formats for matrix multiplications and convolutions natively inside `float32` workflows without changing any code structure. It is often turned off by default for strict backward compatibility.

* **Impact:** Speeds up execution speeds by up to 2x to 3x on convolutional layers with zero modification to the training pipeline loop logic.  
* **Implementation:** Place these lines at the very top of the execution script:

Python

import torch

torch.backends.cuda.matmul.allow\_tf32 \= True

torch.backends.cudnn.allow\_tf32 \= True

### **3\. Page-Locked Memory & Non-Blocking Transfers**

While `num_workers=2` multi-threads data extraction, the transfer of that data from system host RAM to GPU device memory can still create micro-stalls. Enabling memory pinning allows the operating system to pass data directly via DMA (Direct Memory Access), bypassing the CPU.

* **Impact:** Smooths out host-to-device bottlenecks, ensuring the GPU is never waiting for a patch data tensor to clear the PCIe bus lanes.  
* **Implementation:** \`\`\`python

# **In your DataLoader setup**

train\_loader \= DataLoader(

train\_ds,

batch\_size=2,

num\_workers=2,

pin\_memory=True \# Page-locks host RAM for direct GPU access

)

# **In your training loop**

inputs \= batch\_data\["image"\].to(device, non\_blocking=True)

labels \= batch\_data\["label"\].to(device, non\_blocking=True)

\#\#\# 4\. 3D Channels-Last Memory Format

By default, PyTorch structures tensors in memory as NCDHW (Batch, Channels, Depth, Height, Width). However, standard hardware processing arrays operate significantly faster when channels are stored contiguously in memory via the NDHWC layout (Channels-Last).

\* \*\*Impact:\*\* Maximizes memory bandwidth utilization during 3D spatial convolutions.

\* \*\*Implementation:\*\*

\`\`\`python

\# Convert model to channels-last 3D layout

model \= model.to(device, memory\_format=torch.channels\_last\_3d)

\# Inside the training loop, convert inputs

inputs \= inputs.to(device, memory\_format=torch.channels\_last\_3d, non\_blocking=True)

To help visualize how stacking these specific performance variables alters processing throughput, computational load, and potential hardware bottlenecks, use the simulation tool below to analyze your training layout configurations.

### **5\. Algorithmic Convergence: OneCycleLR**

If the objective is reducing total training time, reducing the time *per epoch* is only half the battle; the other half is reducing the *total number of epochs* required to converge. Switching from standard linear decay to a **OneCycle Learning Rate Policy** forces the model to train at higher learning rates during the middle of the regime, regularizing the network and dramatically cutting total optimization loops.

Which of these pipeline configurations (AMP, TF32, or Pinned Memory) are you planning to integrate first into the script architecture?

