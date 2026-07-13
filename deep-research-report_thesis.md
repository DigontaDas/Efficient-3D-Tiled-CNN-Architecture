# Deep Research Report on Your P3 Coronary CCTA Thesis Stack

## Thesis motive from your poster

From the poster you shared, your thesis is not just “another segmentation project.” The real motive is a clinically usable pipeline for **coronary artery stenosis detection and analysis from 3D CCTA**, built to be computationally efficient enough for practical deployment. The poster shows four benchmark model families already in scope—**3D U-Net, V-Net, SegResNet, and nnU-Net v2**—and your intended direction is a **hybrid residual U-Net with deep supervision**, pre-trained on a public secondary dataset and then adapted to a local, radiologist-refined dataset. It also shows that you care about more than masks: you want a path from segmentation to **stenosis quantification, calcification-related analysis, and clinically interpretable output**.

Based on the “ongoing work” panel, I am reading your current **P3** as the stage where you:  
**build or refine the local dataset, generate and correct initial masks, transfer-learn from the public data, and begin training the hybrid model on local CCTA.** That interpretation fits what you wrote here as well: you already have a secondary dataset, and now you want the repos that matter most for training, testing, and making the new local dataset.

## What P3 should focus on now

The biggest strategic decision for P3 is this: **do not start by chasing a novel architecture repo first**. Start by making the **dataset and benchmark loop rock-solid**, then use your custom hybrid model as a controlled improvement over a strong baseline. That is especially important because the latest nnU-Net benchmarking work argues that many newer 3D segmentation claims do not hold up under rigorous validation, and that well-configured CNN U-Net pipelines inside the nnU-Net framework remain extremely strong baselines. citeturn13view1turn20search0turn23view3

For your case, two public resources matter immediately. **ImageCAS** is still one of the strongest public resources for coronary artery segmentation in CTA: its paper emphasizes that prior public datasets were usually only tens of cases, while ImageCAS provides a large-scale dataset and benchmark with an official split for fair comparison. The official repository also warns that some earlier folders were duplicate cases and that a cleaned version was later provided, so verifying that you are using the corrected release is not optional. citeturn14view1turn17view0

For external generalization, **ASOCA** is worth adding if licensing and access are feasible. It is the official “Automated Segmentation of Coronary Arteries” challenge dataset for normal and diseased coronary arteries, and recent reviews still describe ASOCA as the most widely used benchmark in this niche. That matters because patient-level splitting and external validation are both central to avoiding leakage and inflated performance in medical imaging studies. citeturn15search0turn15search2turn15search13turn11search0turn11search2turn11search4

So, in practical terms, P3 should aim for this evaluation logic: **pretrain on ImageCAS, fine-tune on your local hospital data, keep patient-wise local train/validation/test separation, and—if possible—add ASOCA as an external test of generalization.** That design matches your poster’s transfer-learning direction and also follows current methodological best practice in medical imaging evaluation. citeturn17view0turn15search0turn11search0turn11search4turn11search23

## Repositories you should make your core stack

If I reduce everything to the repos that will genuinely carry the project, the **core stack** is quite small.

**MIC-DKFZ/nnUNet** should be your baseline engine and probably your first serious training repo. It is a self-configuring biomedical segmentation framework that covers preprocessing, model configuration, training, postprocessing, and ensembling. nnU-Net v2 supports multiple file formats, not just `.nii.gz`, and its docs explicitly note support for `.nii.gz`, `.nrrd`, `.mha`, and more through reader/writer backends. It also gives you integrity checking, automatic dataset fingerprinting, 5-fold cross-validation, and newer residual-encoder presets scaled to different GPU budgets. In your setting, that makes nnU-Net the best “truth baseline” before you claim gains from a custom hybrid architecture. citeturn20search0turn23view0turn23view2turn23view3turn16view0turn13view1

**Project-MONAI/MONAI** should be your custom-model laboratory. MONAI extends PyTorch with medical-imaging-specific transforms, architectures, utilities, and training workflows. This is the repo you want when the time comes to implement your poster’s hybrid idea—residual blocks, deep supervision, custom losses, or mixed pipelines that do not fit neatly inside nnU-Net defaults. MONAI is also the most natural place to code a controlled version of your thesis model while keeping preprocess/augment/evaluation logic reproducible. citeturn13view2turn8view2turn16view1

**Project-MONAI/tutorials** matters almost as much as MONAI itself, because it gives you the practical recipes you will actually reuse. In particular, **Auto3DSeg** is useful for quickly generating competitive segmentation bundles with minimal user input, and it can ensemble the best checkpoints across folds and algorithms. For P3, that means you can get a fast, strong “engineering benchmark” before hand-tuning your hybrid architecture. It is also valuable because Auto3DSeg supports bring-your-own-algorithm workflows, which aligns well with thesis experimentation. citeturn24view1turn24view0turn6search7

**Project-MONAI/MONAILabel** is one of the most important repos for your dataset-making phase. It is specifically built for AI-assisted annotation and integrates with **3D Slicer**. The installation docs show that it supports local Windows and Ubuntu workflows with Python, PyTorch, and CUDA, and that the 3D Slicer plugin can be installed directly through Slicer’s Extension Manager. That makes it a very strong fit for your “generate initial mask, then radiologist refines to gold standard” workflow from the poster. citeturn8view3turn18view0

**Slicer/Slicer** should be your main annotation and visual QA environment. It is a mature, cross-platform open-source platform for visualization and image analysis, with a very large codebase and ecosystem. For thesis work like yours, Slicer is not just a viewer; it is the place where clinicians can inspect 3D volumes, correct leaks, verify distal branches, and review cases consistently. citeturn12view0turn16view2

**vmtk/SlicerExtension-VMTK** is the repo that most directly supports the “stenosis analysis” half of your thesis. This extension for 3D Slicer includes vessel tree segmentation, centerline extraction, cross-section analysis, targeted artery segmentation workflows, stenosis measurement in 1D/2D/3D, and even an arterial calcification pre-processor. If your thesis is meant to move from segmentation masks toward vessel geometry and stenosis quantification, this is one of the most practically useful repos in the whole stack. citeturn12view1turn19view1turn19view2turn19view3

**XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation-based-on-CT** is the official public benchmark repo you should anchor your comparison against. It gives you the benchmark identity, official split, and the exact public reference point your thesis needs. Even though it is not the repo you will build on most heavily day to day, it is one of the most important repos for making your results comparable and defensible. citeturn17view0turn14view1

There are also two support repos I would treat as part of your day-one environment rather than “nice extras.” **SimpleITK/SimpleITK** is excellent for geometry-safe loading, resampling, spacing/origin checks, and scripting image I/O in Python, while **rordenlab/dcm2niix** is the workhorse for converting raw DICOM series into research-friendly formats when needed. Together they solve a lot of silent preprocessing pain before it reaches training. citeturn22view0turn22view1

## Repositories to use selectively

There are several coronary-specific or modern-prompting repos that are useful, but I would not make them the foundation of P3.

**MIC-DKFZ/nnInteractive** is very promising for annotation acceleration. The repo describes itself as a Python backend for state-of-the-art 3D promptable segmentation, trained on 120+ diverse 3D datasets and designed for code-based workflows. For Antigravity and Codex-style iterative scripting, that makes it a better fit than many GUI-only prompt tools. I would use it to generate candidate masks or difficult-region corrections, not as the primary scientific benchmark. citeturn12view2turn20search3turn20search19

**uni-medical/SAM-Med3D** is also worth knowing, especially if you want foundation-model-assisted mask bootstrapping. The official repo provides the model, released data, and a substantial community footprint. Still, I would treat it as an annotation accelerator rather than the backbone of your thesis, because promptable foundation models can vary heavily across medical datasets and still require strict human verification in domain-shifted settings. citeturn12view3turn9search9turn11academia30turn11search2

**TorchIO-project/torchio** is a highly useful pure-Python support repo for loading, preprocessing, augmentation, and patch-based sampling of 3D medical images. If you build the hybrid thesis model in MONAI, TorchIO is one of the best supporting libraries for dataset transforms, sampling, and prototyping, especially in small-data or vessel-focused patch workflows. citeturn12view5turn10search14

For **coronary-specific research code**, I would divide the repos into “reference for ideas” versus “production foundation.”  
**FrancescoLeni/CoronarySegmentation** is a good *ideas repo* because it explicitly explores centerline priors, flexible 2D/3D workflows, and anatomically guided cropping/attention. That makes it very relevant to your stenosis-oriented problem. But I would still keep it in the reference bucket rather than the core stack. citeturn21view0

**WeiliJiang/Ori_Net** is also a reference repo, especially if vessel connectivity becomes one of your hardest failure modes. It is explicitly built to improve connectivity of coronary segmentation in CCTA and includes centerline-direction logic. The reason I would not make it a foundation repo is that the published environment is older—PyTorch 1.7, TorchIO ≤ 0.18.20, Python ≥ 3.6—and the repository itself is comparatively small. citeturn21view1

**chenzhao2023/AGFA** is similarly useful for reading and possibly borrowing preprocessing ideas, but it looks more like a narrow paper-code release than a robust framework. Its own README describes custom renaming, data packing into `.pth` chunks, and a relatively bespoke training flow. That is not what I would want as the heart of a new thesis codebase. citeturn21view2

**YueZhang1029/LIS-Net** is conceptually relevant because semi-supervised coronary segmentation is attractive when local labels are scarce. But the repo explicitly says the code is being gradually opened after article acceptance, and the public footprint is still quite small. That makes it more of a paper trail than a dependable working base right now. citeturn8view6turn9search18

**qurAI-amsterdam/cardiacSegmentationCCTA** is worth keeping in mind if you later want a whole-heart ROI extractor to localize the field before coronary work. It provides training and inference for a 3D CNN on CCTA, but it is whole-heart rather than coronary-specific, expects `.mhd` data, and is much smaller than the major framework repos. I would classify it as optional, useful for ROI thinking, not central to P3. citeturn8view5

**ales-git/DeepCADRADS** is not a P3 priority, but it may become useful after P3 if you extend into stenosis severity or CAD-RADS style downstream classification. The repo is about a clinically inspired CAD-RADS scoring pipeline built from straightened MPR projections and a fine-tuned MaxViT model, using labels that fit clinical routine. Very interesting later; not the repo I would start with now while your core problem is still 3D artery segmentation and local dataset construction. citeturn21view3

**Advanced-AI-in-Medicine-and-Physics-Lab/SynCAS** is intriguing if you later need synthetic-data augmentation. It uses ImageCAS templates, TotalSegmentator-derived anatomy, and an nnU-Net-style training setup to generate synthetic data. But because it focuses on synthetic NCCT generation rather than directly giving you a cleaner CCTA coronary segmentation pipeline, I would treat it as a later-stage augmentation experiment, not an early P3 dependency. citeturn21view4

## A practical P3 workflow for Antigravity and Codex IDE

For a code-first environment like Antigravity and Codex IDE, the cleanest setup is to split your tooling into **three layers**.

The first layer is **data ingestion and QC**. Use **SimpleITK** for geometry validation and batch scripting, and use **dcm2niix** only when you actually need DICOM conversion. Since nnU-Net v2 already supports multiple input formats, you do not need to over-convert everything if your chosen format is already stable and lossless. What you do need is strict consistency of geometry and channels across image-label pairs. citeturn22view0turn22view1turn23view0

The second layer is **dataset creation and clinician-in-the-loop refinement**. Here the physical workflow should be **3D Slicer + MONAI Label + SlicerVMTK**. MONAI Label gives you AI-assisted prelabels and an active-learning style annotation loop; Slicer gives your radiologists and reviewers the working UI; VMTK gives you centerlines, cross-sectional measurement, stenosis modules, and calcification-oriented preprocessing. This trio maps almost perfectly to the workflow described on your poster. citeturn18view0turn12view0turn19view1turn19view2

The third layer is **training and experiments**. Start with **nnU-Net** as the benchmark baseline because it will automatically fingerprint the dataset, verify integrity, run 5-fold cross-validation, and give you a strong reference point. Then build your custom hybrid model in **MONAI**, where you can explicitly control residual blocks, deep supervision, auxiliary losses, and ensemble logic. If annotation remains the biggest bottleneck, add **nnInteractive** or **SAM-Med3D** as optional pre-labeling assistants. citeturn23view2turn24view1turn24view0turn12view2turn12view3

In other words, for your IDE workflow, I would keep the code repo responsibilities very clear. **nnU-Net** is for “how strong is a disciplined baseline on our exact data.” **MONAI** is for “how do we implement the thesis model cleanly.” **MONAI Label / Slicer / VMTK** are for “how do we create trustworthy labels and extract clinically meaningful vessel measurements.” That separation will save you a lot of time and prevent the codebase from turning into a patchwork of unrelated paper repos. citeturn20search0turn13view2turn8view3turn19view1

## Bottom line

If I had to tell you, as directly as possible, **which repos will be in greatest use for your work right now**, my shortlist would be this:

**Install and use immediately:**  
**nnUNet**, **MONAI**, **MONAILabel**, **3D Slicer**, **SlicerExtension-VMTK**, **ImageCAS official repo/data**, with **SimpleITK** and **dcm2niix** as your support utilities. These are the repos that best cover your current P3 needs: local dataset construction, annotation refinement, benchmark training, custom hybrid-model development, and downstream vessel/stenosis analysis. citeturn20search0turn13view2turn8view3turn12view0turn12view1turn17view0turn22view0turn22view1

**Add next, if annotation speed is hurting you:**  
**nnInteractive** first, then **SAM-Med3D** if you want a second promptable-segmentation option. For pure Python augmentation and patch sampling, add **TorchIO**. citeturn12view2turn12view3turn12view5

**Use mainly as reference repos, not as your foundation:**  
**CoronarySegmentation**, **Ori_Net**, **AGFA**, **LIS-Net**, **cardiacSegmentationCCTA**, **DeepCADRADS**, and **SynCAS**. They are valuable for ideas, ablations, or later-stage extensions, but they should not replace the stronger framework-based stack above. citeturn21view0turn21view1turn21view2turn8view6turn8view5turn21view3turn21view4

The most important research judgment, after reading both your poster and the current public ecosystem, is this: **for P3, your best move is not to collect many coronary paper repos, but to combine one rigorous training baseline, one flexible custom-model framework, and one high-quality annotation stack.** In your case, that means **nnU-Net + MONAI + Slicer/MONAI Label/VMTK**. That stack is the one most likely to move your thesis forward quickly and credibly. citeturn13view1turn13view2turn8view3turn19view1