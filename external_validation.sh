# copy feature and outcome files to validation folders
echo 'COPY feature and outcome files to validation folders'

cp X.csv validation/performance_validation/X.csv
cp y_death.csv validation/performance_validation/y_death.csv
cp y_poor_outcome.csv validation/performance_validation/y_poor_outcome.csv

cp X.csv validation/risk_stratification/X.csv
cp y_death.csv validation/risk_stratification/y_death.csv
cp y_poor_outcome.csv validation/risk_stratification/y_poor_outcome.csv

mkdir validation_output

# Get external evaluation performance matrics

echo 'START performance validation'

cd validation/performance_validation

python3 validation_kch.py estimator_DL_death X.csv y_death.csv
python3 validation_kch.py estimator_DL_poor_outcome X.csv y_poor_outcome.csv

cd ../..

mv validation/performance_validation/*.o validation_output/

echo 'DONE performance validation'

# Get risk stratification results

echo 'START risk stratification'

cd validation/risk_stratification
bash make_figures.sh

echo 'DONE risk stratification'

cd ../..
mv validation/risk_stratification/*.svg validation_output/
mv validation/risk_stratification/*.o validation_output/

# delete copied files
rm validation/performance_validation/X.csv
rm validation/performance_validation/y_death.csv
rm validation/performance_validation/y_poor_outcome.csv

rm validation/risk_stratification/X.csv
rm validation/risk_stratification/y_death.csv
rm validation/risk_stratification/y_poor_outcome.csv

echo 'DELETED copied files'


# move output

echo 'SAVED output to validation_output/'

mv validation/performance_validation/*.o validation_output/

mv validation/risk_stratification/*.svg validation_output/
mv validation/risk_stratification/*.o validation_output/

