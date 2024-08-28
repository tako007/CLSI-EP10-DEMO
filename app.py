import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import streamlit as st
import statsmodels.api as sm

# FUNCTIONS
def coefs_(df, prediction=False, summary=False, coef=False):
    if summary:
        y = df["VALUE"]
        X = sm.add_constant(df[["X", "Xsq_adj", "Carryover", "Time"]], prepend=False)
        mod = sm.OLS(y, X)
        res = mod.fit()
    else:
        model = LinearRegression().fit(
            df[["X", "Xsq_adj", "Time", "Carryover"]].to_numpy(),
            df["VALUE"].to_numpy())
        if prediction:
            return pd.Series(model.predict(df[["X", "Xsq_adj", "Time", "Carryover"]]))
        elif coef:
            return np.append(model.coef_, model.intercept_)
def coef_t(df,assigned_mid,assigned_low,run):
    y = df[df.RUNS == run]["VALUE"]
    X = sm.add_constant(df[df.RUNS == run][["X","Xsq_adj","Carryover","Time"]],prepend=False)
    result = sm.OLS(y, X).fit()
    scale_factor = assigned_mid - assigned_low

    B1 = result.params["X"] / scale_factor
    B0 = result.params["const"] - (assigned_mid * B1)
    B2 = (result.params["Carryover"] / result.params["X"]) * 100
    B3 = result.params["Xsq_adj"]  / (scale_factor**2)
    B4 = result.params["Time"]


    bse_B0 = result.bse["const"]
    bse_B1 = result.bse["X"] / scale_factor
    bse_B2 = result.bse["Carryover"]
    bse_B3 = result.bse["Xsq_adj"] / (scale_factor**2)
    bse_B4 = result.bse["Time"]


    t_B0 = B0 / bse_B0
    t_B1 = (B1 - 1) / bse_B1
    t_B2 = result.params["Carryover"] / bse_B2
    t_B3 = B3 / bse_B3
    t_B4 = result.params["Time"] / bse_B4

    return pd.DataFrame([[B0,B1,B2,B3,B4],
    [bse_B0,bse_B1,bse_B2,bse_B3,bse_B4],
    [t_B0,t_B1,t_B2,t_B3,t_B4]]).T.set_axis(["adj_coef","std_error","t_value"],axis="columns")
# INITIALIZE SESSION_STATES
def initialize_session_state(**kwargs):
    for key, value in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = value

def check_data_validity(*dataframes):
    for df in dataframes:
        if (df <= 0).any().any():
            return False
    return True

st.markdown("<h1 style='text-align: center;'>CLSI EP10 DEMONSTRATION</h1>", unsafe_allow_html=True)
# INITIAL DATA
initial_data_runs = pd.DataFrame([[92,92,93,90,92],
[9 ,9,9,9,9],
[54,54,54,54,52],
[56,54,54,55,55],
[10,9,9,9,9],
[9 ,8,9,9,9],
[92,91,92,92,92],
[95,92,96,94,94],
[59,56,58,52,53]], columns=[f'Run{i+1}' for i in range(5)], index=[ "High", "Low", "Mid", "Mid", "Low", "Low", "High", "High", "Mid"])
initial_data_assigned = pd.DataFrame([[9.0, 50.5, 92.0]], columns=["Low","Mid","High"], index=["assigned"])
initial_data_allow_bias =  pd.DataFrame([[2,4,5]], columns=["Low","Mid","High"], index=["Allowable bias"])
initial_data_allow_imprecision = pd.DataFrame([[8,3,2]], columns=["Low","Mid","High"], index=["Allowable imprecision(%)"])


initialize_session_state(runs=initial_data_runs, 
                         assigned=initial_data_assigned, 
                         allowi = initial_data_allow_imprecision, 
                         allowb = initial_data_allow_bias,
                         show_imprecision = True,
                         show_bias = True,
                         show_mr = True,
                         detailed_mr = False)

# ------- DATA EDITOR -------
st.markdown("<i>Enter your data here:</i>", unsafe_allow_html=True)
st.session_state.runs = st.data_editor(st.session_state.runs,use_container_width=True)
st.markdown("<i>Enter assigned values for your samples here:</i>", unsafe_allow_html=True)
st.session_state.assigned = st.data_editor(st.session_state.assigned, use_container_width=True)
st.markdown("<i>Enter allowable imprecision for your data here:</i>", unsafe_allow_html=True)
st.session_state.allowi = st.data_editor(st.session_state.allowi, use_container_width=True)
st.markdown("<i>Enter allowable bias for your data here:</i>", unsafe_allow_html=True)
st.session_state.allowb = st.data_editor(st.session_state.allowb, use_container_width=True)

data_validity = check_data_validity(st.session_state.runs, st.session_state.assigned, st.session_state.allowi, st.session_state.allowb)

# DESTRUCTURING DATA EDITOR
assigned_low, assigned_mid, assigned_high = st.session_state.assigned.stack().to_numpy().round(2)
allowable_imprecision_low, allowable_imprecision_mid, allowable_imprecision_high = st.session_state.allowi.stack().to_numpy().round(2)
allowable_bias_low, allowable_bias_mid, allowable_bias_high = st.session_state.allowb.stack().to_numpy().round(2)


# WIDE TO LONG DATA
data = st.session_state.runs.copy()
data["SEQUENCE NUMBER"] = range(1,10)
data = data.set_index("SEQUENCE NUMBER",append=True)
data = data.stack().reset_index().set_axis(["LEVEL","SEQUENCE NUMBER","RUNS","VALUE"],axis="columns")
data = data.sort_values(["RUNS","SEQUENCE NUMBER"]).reset_index(drop=True)
coef_data = pd.concat([pd.DataFrame(np.array([[1, 1.0, 0.333333, -4.0, 0.0, 1.0],
[-1, -1.0, 0.333333, -3.0, 1.0, 1.0],
[0, 0.0, -0.66667, -2.0, -1.0, 0.0],
[0, 0.0, -0.66667, -1.0, 0.0, 0.0],
[-1, -1.0, 0.333333, 0.0, 0.0, 1.0],
[-1, -1.0, 0.333333, 1.0, -1.0, 1.0],
[1, 1.0, 0.333333, 2.0, -1.0, 1.0],
[1, 1.0, 0.333333, 3.0, 1.0, 1.0],
[0, 0.0, -0.66667, 4.0, 1.0, 0.0]]))]*5).reset_index(drop=True).set_axis(["CODED VALUES","X","Xsq_adj","Time","Carryover","Xsq"],axis="columns")
data = pd.concat([data,coef_data],axis=1).sort_values(["RUNS","SEQUENCE NUMBER"]).reset_index(drop=True)
data = data.assign(assigned = lambda x:x.LEVEL.map(dict(zip(["Low","Mid","High"],[assigned_low, assigned_mid, assigned_high]))),
           diff=lambda x:x.VALUE-x.assigned,
           percent_diff = lambda x:x["diff"]/x.assigned)


# --------VISUAL INSPECTION
if data_validity:
    plot_type = st.radio("Plot type:", ["diff", "percent_diff"], index=0,)
    fig = px.scatter(data, x='assigned', y=plot_type, hover_data=['RUNS'])
    fig.add_hline(y=0, line=dict(color='red'), name='y=0')
    st.plotly_chart(fig)
else:
    st.write("Error: All values in both DataFrames must be greater than 0.")

# BIAS EVALUATION
low = data[data.LEVEL == "Low"]
mid = data[data.LEVEL == "Mid"]
high = data[data.LEVEL == "High"]
low = low.assign(sum = low.groupby("RUNS")["VALUE"].transform("sum"),
        mean = low.groupby("RUNS")["VALUE"].transform("mean").round(2),
        sd = low.groupby("RUNS")["VALUE"].transform("std").round(2),
        var = low.groupby("RUNS")["VALUE"].transform("var").round(2))
mid = mid.assign(sum = mid.groupby("RUNS")["VALUE"].transform("sum"),
        mean = mid.groupby("RUNS")["VALUE"].transform("mean").round(2),
        sd = mid.groupby("RUNS")["VALUE"].transform("std").round(2),
        var = mid.groupby("RUNS")["VALUE"].transform("var").round(2))
high = high.assign(sum = high.groupby("RUNS")["VALUE"].transform("sum"),
        mean = high.groupby("RUNS")["VALUE"].transform("mean").round(2),
        sd = high.groupby("RUNS")["VALUE"].transform("std").round(2),
        var = high.groupby("RUNS")["VALUE"].transform("var").round(2))
mean_sd = pd.concat([low.drop_duplicates(["RUNS"]).set_index("RUNS")[["sd","mean"]].rename(lambda x:"low_"+x,axis=1),
mid.drop_duplicates(["RUNS"]).set_index("RUNS")[["sd","mean"]].rename(lambda x:"mid_"+x,axis=1),
high.drop_duplicates(["RUNS"]).set_index("RUNS")[["sd","mean"]].rename(lambda x:"high_"+x,axis=1)],axis=1)
bias_low, bias_mid, bias_high =  np.array(mean_sd.filter(like="mean").mean().values - np.array([assigned_low, assigned_mid, assigned_high])).round(2)

if st.button("Show/Hide Bias Evaluation"):
    st.session_state.show_bias = not st.session_state.show_bias
if st.session_state.show_bias and data_validity:
    st.write(f"🥳 Bias for low concentration is acceptable (Bias={bias_low} Allowable bias={allowable_bias_low})" 
             if bias_low < allowable_bias_low 
             else f"😱 Bias for low concentration is unacceptable (Bias={bias_low} Allowable bias={allowable_bias_low})")
    st.write(f"🥳 Bias for middle concentration is acceptable (Bias={bias_mid} Allowable bias={allowable_bias_mid})" 
             if bias_mid < allowable_bias_mid 
             else f"😱 Bias for middle concentration is unacceptable (Bias={bias_mid} Allowable bias={allowable_bias_mid})")
    st.write(f"🥳 Bias for high concentration is acceptable (Bias={bias_high} Allowable bias={allowable_bias_high})" 
             if bias_high < allowable_bias_high 
             else f"😱 Bias for low concentration is unacceptable (Bias={bias_high} Allowable bias={allowable_bias_high})")


V_w = ((mean_sd.filter(like="sd") ** 2).sum() / 5).round(2).to_numpy()
V_m = np.array([low.groupby("RUNS")["VALUE"].mean().var(), mid.groupby("RUNS")["VALUE"].mean().var(), high.groupby("RUNS")["VALUE"].mean().var()])
V_d =  (V_m - V_w/3).round(3)
V_d = np.where(V_d < 0, 0, V_d)
V_t = V_w + V_d
S = V_t ** .5
S = S.round(2)
Y = mean_sd.filter(like="mean").mean().round(1).to_numpy()
C = ((S / Y) * 100).round(2)
imprecision_low, imprecision_mid, imprecision_high = list(C)
if st.button("Show/Hide Imprecision Evaluation"):
    st.session_state.show_imprecision = not st.session_state.show_imprecision
if st.session_state.show_imprecision and data_validity:
    st.write(f"🥳 Imprecision for low concentration is acceptable (Imprecision={imprecision_low} Allowable imprecision={allowable_imprecision_low})" 
             if imprecision_low < allowable_imprecision_low 
             else f"😱 Imprecision for low concentration is unacceptable (Imprecision={imprecision_low} Allowable imprecision={allowable_imprecision_low})")
    st.write(f"🥳 Imprecision for middle concentration is acceptable (Imprecision={imprecision_mid} Allowable imprecision={allowable_imprecision_mid})" 
             if imprecision_mid < allowable_imprecision_mid 
             else f"😱 Imprecision for middle concentration is unacceptable (Imprecision={imprecision_mid} Allowable imprecision={allowable_imprecision_mid})")
    st.write(f"🥳 Imprecision for high concentration is acceptable (Imprecision={imprecision_high} Allowable imprecision={allowable_imprecision_high})" 
             if imprecision_high < allowable_imprecision_high 
             else f"😱 Imprecision for low concentration is unacceptable (Imprecision={imprecision_high} Allowable imprecision={allowable_imprecision_high})")


# MULTIPLE REGRESSION EVALUATION
slope_coef = [139, -96, 11, 8, -117, -126, 100, 100, -19]
carryover_coef = [26, 130, -102, 8, -4, -126, -126, 100, 94]
nonlinearity_coef = [87, 96, -237, -234, 117, 126, 126, 126, -207]
drift_coef = [-52, -34, -22, -16, 8, 26, 26, 26, 38]
data = data.assign(slope_coef=np.tile(slope_coef, data.RUNS.nunique()),
                    carryover_coef=np.tile(carryover_coef, data.RUNS.nunique()),
                    nonlinearity_coef=np.tile(nonlinearity_coef, data.RUNS.nunique()),
                    drift_coef=np.tile(drift_coef, data.RUNS.nunique()))
data = data.assign(slope_subtotal=data.VALUE * data.slope_coef,
                    carryover_subtotal=data.VALUE * data.carryover_coef,
                    nonlinearity_subtotal=data.VALUE * data.nonlinearity_coef,
                    drift_subtotal=data.VALUE * data.drift_coef)
data["predicted"] = data.groupby("RUNS").apply(coefs_).T.stack().sort_index(level=1).reset_index(drop=True)
data["residual"] = data.VALUE - data.predicted
data["residual_sq"] = ((data.VALUE - data.predicted)**2).round(2)
df_t = pd.concat([(abs(coef_t(data, assigned_mid, assigned_low, day).t_value) > 4.6).rename(day + " t Value") 
                  for day in data.RUNS.unique().tolist()], axis=1).replace({True: "X", False: "✓"})
df_t.index = ["Intercept (B0adj)", "Slope (B1adj)", "%Carryover (B2adj)", "Nonlinearity (B3adj)", "Drift (B4)"]

if st.button("Show/Hide Multiple Regression Evaluation"):
    st.session_state.show_mr = not st.session_state.show_mr
if st.session_state.show_mr and data_validity:
    st.dataframe(df_t,use_container_width=True)

    if st.button("Show/Hide Detailed Multiple Regression Results"):
        st.session_state.detailed_mr = not st.session_state.detailed_mr
    if st.session_state.detailed_mr:
        st.dataframe(pd.concat([coef_t(data, assigned_mid, assigned_low, day)[["adj_coef", "t_value"]].T
                                .set_axis(["Intercept (B0adj)", "Slope (B1adj)", "%Carryover (B2adj)", "Nonlinearity (B3adj)", "Drift (B4)"], axis="columns")
                                .assign(index=[day + " adj_coef", day + " t_value"]).set_index("index")
                                for day in data.RUNS.unique().tolist()]).round(2)
                     .style.map(lambda val: 'background-color: red' if abs(val) > 4.6 else '').format("{:.2f}")
, use_container_width=True)
