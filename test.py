"""import os

path="notebooks/research.ipynb"

dir,file=os.path.split(path)

os.makedirs(dir,exist_ok=True)

with open(path,"w") as f:
    pass"""


from src.Diamondpriceprediction.pipelines.prediction_pipeline import CustomData, PredictPipeline

data = CustomData(1.52, 62.2, 58.0, 7.27, 7.33, 4.55, "Premium", "F", "VS2")

print(data)

final_data = data.get_data_as_dataframe()

print(final_data)

predict_pipeline = PredictPipeline()
pred = predict_pipeline.predict(final_data)
print(type(pred), pred)
result = round(pred[0], 2)
print("result:", result)

# o/p
# DELL@SachinKPC MINGW64 /d/Mytech/IDE_workspace/ksachin5136git/DiamondPricePredictionProject_repo/DiamondPricePredictionProject (main)
# $ python ./test.py
# <src.Diamondpriceprediction.pipelines.prediction_pipeline.CustomData object at 0x000001D0C9EDD0A0>
#    carat  depth  table     x     y     z      cut color clarity
# 0   1.52   62.2   58.0  7.27  7.33  4.55  Premium     F     VS2
# <class 'numpy.ndarray'> [10983.24194409]
# result: 10983.24
