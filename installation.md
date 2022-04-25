Create a folder for Github project.

Clone the main branch to main folder

git clone https://github.com/fusumwan/aistockprediction.git

cd /..[your directionary]../aistockprediction

git branch -M main
git remote add origin https://github.com/fusumwan/aistockprediction.git
git push -u origin main





The main function is "MLPrediction.py", user need to type the following command to run this program. 


python3 MLPrediction.py;


This command will load both "stock.csv" and "stock_volume.csv" files and then it will generate different chat graphics for different stock. At least it will export the "Result1.txt" which include all accuracy results.


There many graphics are exported into several folders, such as "Average","Ridge","LSTM", and "Original". For instance, the stock price charts of the 9 companies from January 12, 2012 to August 11, 2020 are all placed in the "original" folder. The prediction results of the Ridge regression model and LSTM model will be saved in "Ridge" and "LSTM" respectively. Finally, the "average" folder contains all the prediction results using the effective averaging method.

Need to install some package: pip install scipy, pip3 install scipy, pip install plotly,pip3 install plotly,pip install numpy,pip3 install numpy,python -m pip install -U pip ,python -m pip install -U matplotlib,python3 -m pip3 install -U pip,python3 -m pip3 install -U matplotlib,pip install pandas,pip3 install pandas,pip install --upgrade tensorflow,pip3 install --upgrade tensorflow,pip install scikit-learn,pip3 install scikit-learn,pip install -U kaleido,pip3 install -U kaleido