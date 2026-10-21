# Data folder

Populate this folder as follows (see scripts/ for how to get the images):

```
data/
├── mango/
│   ├── train/<ClassName>/*.jpg
│   └── val/<ClassName>/*.jpg
├── neem/
│   ├── train/<ClassName>/*.jpg
│   └── val/<ClassName>/*.jpg
└── tamarind/
    ├── train/<ClassName>/*.jpg
    └── val/<ClassName>/*.jpg
```

`<ClassName>` folders are auto-detected by `torchvision.datasets.ImageFolder`
based on their names — whatever you name these folders becomes your model's
output classes. Make sure they match the keys in `treatment_db.json` for
that plant (or update treatment_db.json to match your folder names).
