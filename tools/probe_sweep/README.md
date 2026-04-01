# ManifoldProbe: Data-Free KD via Saliency-Guided Pixel Sweep

## Idea
Teacher의 gradient saliency로 영향력 큰 픽셀을 찾고,
해당 픽셀을 0~255로 전수조사하여 synthetic dataset을 만들어 Student를 학습시킨다.

## Directory Structure (새로 추가되는 파일들)

```
mdistiller/
├── mdistiller/
│   ├── models/
│   │   └── mnist.py              # TeacherCNN, StudentCNN
│   ├── dataset/
│   │   ├── mnist.py              # MNIST dataloader
│   │   └── synthetic.py          # Synthetic sweep dataset
│   └── distillers/
│       └── ManifoldProbe.py      # Saliency probe + KD distiller
└── tools/
    └── probe_sweep/
        ├── README.md             # This file
        ├── step1_train_teacher.py
        ├── step2_saliency.py
        ├── step3_pixel_sweep.py
        ├── step4_distill.py
        └── step5_evaluate.py
```

## How to Run

```bash
cd mdistiller

# Step 1: Teacher 학습 (~2분, GPU)
python tools/probe_sweep/step1_train_teacher.py

# Step 2: Saliency 추출 (~1분)
python tools/probe_sweep/step2_saliency.py --top_k 20

# Step 3: Pixel sweep으로 synthetic data 생성 (~5분)
python tools/probe_sweep/step3_pixel_sweep.py --top_k 20 --num_values 256 --seed_mode mean

# Step 4: Student distillation (~3분)
python tools/probe_sweep/step4_distill.py --temperature 4.0 --epochs 30

# Step 5: 전체 비교 평가
python tools/probe_sweep/step5_evaluate.py
```

## Outputs

```
checkpoints/
├── teacher_mnist.pth         # Teacher checkpoint
├── student_manifold.pth      # Our method
├── student_real.pth          # Baseline: student on real data
└── student_random.pth        # Baseline: student on random data

outputs/
├── saliency/
│   ├── all_classes.pt        # Per-class saliency maps + pixel lists
│   ├── class_0.pt ~ class_9.pt
│   └── saliency_overview.png # Visualization
└── sweep_dataset.pt          # Synthetic dataset (~50K pairs)
```

## Experiment Grid

| Exp | Training Data          | Expected Acc |
|-----|------------------------|-------------|
| A   | Teacher (real MNIST)   | ~99%        |
| B   | Student (real MNIST)   | ~98.5%      |
| C   | Student (ManifoldProbe)| ?           |
| D   | Student (random noise) | ~10-20%     |

**성공 기준**: C >> D (saliency sweep이 random보다 유의미하게 좋음)

## Key Hyperparameters

- `top_k`: 클래스당 sweep할 픽셀 수 (default: 20)
- `num_values`: 픽셀당 sweep 단계 수 (default: 256)
- `seed_mode`: seed 이미지 전략 (mean/first/random)
- `temperature`: KD temperature (default: 4.0)
