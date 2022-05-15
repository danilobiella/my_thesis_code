
plots/final_bounds.png : data/1232/interim/template.ascii
	python find_bounds.py --lc_file data/nuova_obs/raw/licutot18.ascii --niter 1 --template_in data/1232/interim/template.ascii

data/1232/interim/template.ascii :
	python find_bounds.py --lc_file data/1232/raw/lightcurve1tot.txt --niter 5 --template_out data/1232/interim/template.ascii
