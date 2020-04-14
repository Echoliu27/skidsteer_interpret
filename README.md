# skidsteer_interpret
## How to run this demo

- [Demo app](https://skidsteer-interpret.herokuapp.com/)

- Run from repo

```
git clone https://github.com/Echoliu27/skidsteer_interpret.git
cd skidsteer_interpret
make install
streamlit run app.py
```
## File Structure
```bash
├── data
│   ├── brand_influence.csv
│   ├── feature_importance.csv
│   ├── final_tabular2.csv            ## standardized and log transformed data (except winning_bid)
│   ├── final_unscaled.csv            ## unscaled data
│   ├── mturk_96.csv                  ## mturk data
├── image
│   ├── rf_image
│   │   ├── bigiron_AC6813.jpg        ## original image
│   │   ├── ...
│   └── most_least_colorfulness.PNG   ## demo image for a button
├── results
│   ├── images
│   │   ├── 118_cam.png               ## cam image (118 corresponds to the index/row number in results_val.csv)
│   │   ├── 118_gn.png                ## guided saliency image
│   │   ├── ...
│   ├── file_list.csv                 
│   ├── results_train.csv             ## training set for nn
│   └── results_val.csv               ## validation set for nn
├── app.py
├── requirements.txt
├── Makefile
├── README.md
├── Procfile
├── setup.sh
└── .gitignore
```
## Model 1: Random Forest
- Use dataset (joined with MTURK annotation) with sample size 96 as train and test set
- Prediction accuracy is not as good as neural network due to small sample size
- Local interpretaion shows how each feature collectively influence the prediction outcome using Shapely value.

## Model 2: Pre-trained Neural Net
- **Attention**：This is only a demo with no actual model loaded since model is too large to be uploaded onto Github. 
We stored the images ouput by the model in a folder named results/images and directly fetch pictures to render interpretations.
