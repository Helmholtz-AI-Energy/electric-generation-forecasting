from models.start_forecasts import start_pslp
from models.data_preparation import prepare_data


###########################################################
### Selection options #####################################
###########################################################

forecast_method = "pslp" # select: "pslp", "transformer" ....
mode = "load" # select: "load", "generation" ......
pslp_opt = "fix" # select: "standard", "fix" or "var"
plot_opt = False # plot predictions "True" or "False"
start_date = "2020-06-01"
end_date = "2022-12-31"

###########################################################


###### read data ########

print("prepare " + mode + " data")
data, scores = prepare_data(mode, start_date, end_date)

###### start prediction with personalized standard load profiles ########

if forecast_method == "pslp":
    print("start " + mode + " Personalized Standard Load Profile (PSLP) in forecast mode: " + pslp_opt)
    scores, real, pred = start_pslp(data, forecast_method, pslp_opt, scores, mode, plot_opt)

