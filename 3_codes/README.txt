Run from this folder after: pip install -r requirements.txt

  python3 src/run_physics.py
  python3 src/shear_elasticity.py
  python3 src/make_report_figures.py

src/carreau_pipe.py solves the laminar Carreau pipe flow.
src/run_physics.py fits n and reads the transition Reynolds number.
src/shear_elasticity.py is the shear-rate correction to E(1-beta).
src/digitized_data.py holds the values read from the article.
src/build_dataset.py writes data/friction_factor.csv from those values.
The two csv files already in data/ are the inputs used for the report.
