from models.pslp_forecasts import modelling_pslp, modelling_pslp_fix, modelling_pslp_var


def start_pslp(data, forecast_method, pslp_opt, scores, mode, plot_opt):
    if pslp_opt == "standard":
        scores, pred, real = modelling_pslp(data, 21, 1, scores, forecast_method, pslp_opt, mode, plot_opt) # run the standard pslp prediction
    elif pslp_opt == "fix":
        scores, pred, real = modelling_pslp_fix(data, 21, 1, scores, forecast_method, pslp_opt, mode, plot_opt) # run the fix pslp prediction
    elif pslp_opt == "var":
        scores, pred, real = modelling_pslp_var(data, 21, 1, scores, forecast_method, pslp_opt, mode, plot_opt) # run the variable pslp prediction


    scores.to_csv("./results/"+forecast_method+"/"+mode+"/"+pslp_opt+"/"+ "score_pslp_"+pslp_opt+"_"+mode+".csv", sep=";")
    real.to_csv("./results/"+forecast_method+"/"+mode+"/"+pslp_opt+"/"+ "real_pslp_"+pslp_opt+"_"+mode+".csv", sep=";")
    pred.to_csv("./results/"+forecast_method+"/"+mode+"/"+pslp_opt+"/"+ "pred_pslp_"+pslp_opt+"_"+mode+".csv", sep=";")



    return scores, real, pred
