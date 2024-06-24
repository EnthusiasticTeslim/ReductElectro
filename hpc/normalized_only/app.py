import numpy as np
import pymatgen.core as pmg
import streamlit as st

def cu_fraction(Sn_percent):
    if Sn_percent <= 1:
        Cu = 1 - Sn_percent
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return Cu

def get_weight(Sn_percent):
    # create the structure
    if Sn_percent <= 1:
        base = f'Cu{1-Sn_percent}Sn{Sn_percent}'
        comp = pmg.Composition(base)
    else:
        raise ValueError('Sn percent must be less than or equal to 1')
    return comp.weight

def ethane(Sn_percent, Pot, weight, pH, Cu_percent, cDen):
    c0 = -0.004914360452691078
    a0 = -0.2066222976907354
    a1 = 0.5316870806624426
    a2 = -0.3948118857569158
    a3 = 0.7149525683414504
    a4 = 0.620296271014388
    a5 = -0.3429464114810739
    result = c0 + a0 * abs((Sn_percent / Pot) - (weight**3)) + a1 * abs((pH * Pot) - np.sqrt(cDen)) + a2 * ((pH - Pot) * (Cu_percent / cDen)) + a3 * ((Cu_percent / Pot) - (Cu_percent / pH)) + a4 * ((Cu_percent**6) * (Cu_percent - pH)) + a5 * (np.cbrt(Sn_percent) - np.cos(Cu_percent))

    return result if result > 0 else 0

def carbonmono(Sn, Cu, Pot, pH, cDen):
    a0 = 3.277493220571178e-01
    a1 = -5.598909495281141e-01
    a2 = 9.546056163094251e-01
    a3 = -2.895967612298517e+00
    a4 = 1.204578438049861e-01
    a5 = -3.447184433917704e-01
    c0 = 1.886998352450209e-01
    
    ans = c0 + a0 * (np.abs(np.sqrt(Sn) - np.abs(Cu - Pot))) + a1 * (np.abs((pH - cDen) - (Pot**6))) + a2 * (np.abs((cDen / pH) - np.cos(Pot))) + a3 * ((Sn**6) * (Cu * cDen)) + a4 * (np.abs((Cu / cDen) - (Cu + Pot))) + a5 * (np.abs(np.sqrt(Sn) - (cDen / pH)))
    return ans if ans > 0 else 0


def ethanol(Cu, weight, Pot, cDen, Sn, pH):
    c0 = 2.801280473445035e-01
    a0 = 3.516697023069468e-01
    a1 = 3.687251845185032e-01
    a2 = -5.970932191479567e-01
    a3 = -7.073091663701121e-01
    a4 = -1.018106173107204e-01
    a5 = -2.343047991878564e-01
    
    ans = c0 + a0 * ((abs(Sn - cDen)) * (abs(Cu - cDen))) + a1 * (abs((np.exp(-1.0 * Pot)) - (Cu ** 6))) + a2 * (abs(np.cbrt(Sn) - (weight ** 2))) + a3 * (abs((cDen / Pot) - (pH * Pot))) + a4 * (abs((Cu / cDen) - (weight + Pot))) + a5 * (abs((Cu ** 6) - Pot))
    return ans if ans > 0 else 0

def formate(weight, Sn_percent, Cu_percent, Pot, cDen, pH):
    a0 = -0.1001127428547696
    a1 = -0.09893185561627102
    a2 = -0.2577044948418116
    a3 = 0.3546914036535919
    a4 = -0.2754312170894154
    a5 = -1.318852499278862
    c0 = 0.8093928041557034
    
    ans = c0 + a0 * np.abs(np.abs(Cu_percent - cDen) - np.sqrt(Cu_percent)) + a1 * np.abs((pH * cDen) - np.abs(Cu_percent - Sn_percent)) + a2 * np.abs((Cu_percent**6) - np.exp(-1.0 * weight)) + a3 * np.abs(np.cbrt(Sn_percent) - (weight**2)) + a4 * ((Pot**6)**6) + a5 * np.abs((weight * Sn_percent) - np.abs(Cu_percent - weight))

    return ans if ans > 0 else 0

def hydrogen(Sn, Cu, Pot, pH, cDen, weight):
    a0 = 1.863778473177432e-02
    a1 = 5.910962254915521e+00
    a2 = 4.228183481178416e-02
    a3 = 7.127487610097909e-01
    a4 = -2.265716650451252e-01
    a5 = 1.669467979327007e-01
    c0 = 2.992789562320270e-02

    ans = c0 + a0 * np.abs((cDen**3) - np.abs(Cu - weight)) + a1 * (Sn**2 * Cu**6) + a2 * np.abs((Pot / pH) - np.cbrt(weight)) + a3 * np.abs(np.sin(cDen) - (Pot * cDen)) + a4 * (np.cbrt(Sn) + (Cu - Pot)) + a5 * (np.exp(-1.0 * pH) + cDen**6)
   
    return ans if ans > 0 else 0

def calculate_outputs(Sn_percent, Pot, cDen, pH):
    Cu_percent = cu_fraction(Sn_percent) / 1.00 # max Cu fraction
    print(f'Weight of the structure: {get_weight(Sn_percent)}')
    weight = get_weight(Sn_percent) / 118.71 # max weight of the structure

    Pot /= 4.70 # max potential
    pH /= 14.05 # max pH
    cDen /= 450.00 # max current density
    
    get_ethane = ethane(Sn_percent=Sn_percent, Pot=Pot, weight=weight, pH=pH, Cu_percent=Cu_percent, cDen=cDen)
    get_carbonmono = carbonmono(Sn=Sn_percent, Cu=Cu_percent, Pot=Pot, pH=pH, cDen=cDen)
    get_ethanol = ethanol(Cu=Cu_percent, weight=weight, Pot=Pot, cDen=cDen, Sn=Sn_percent, pH=pH)
    get_formate = formate(weight=weight, Sn_percent=Sn_percent, Cu_percent=Cu_percent, Pot=Pot, cDen=cDen, pH=pH)
    get_hydrogen = hydrogen(Sn=Sn_percent, Cu=Cu_percent, Pot=Pot, pH=pH, cDen=cDen, weight=weight)

    
    return {
        'C2H4 FE': round(get_ethane*100, 2),
        'CO FE ': round(get_carbonmono*100, 2),
        'Ethanol FE': round(get_ethanol*100, 2),
        'Formate FE': round(get_formate*100, 2),
        'H2 FE': round(get_hydrogen*100, 2)

    }


st.title('Electrochemical Reaction Outputs')

st.text('This app calculates the Faradaic efficiency of the electrochemical reactions based on the input parameters.')

st.write('## Input Parameters')
cDen = st.number_input('Current Density (cDen)', min_value=141.00, max_value=450.00, value=200.00)
cDen_slider = st.slider('Adjust Current Density (cDen)', 141.00, 450.00, value=cDen)

Pot = st.number_input('Potential (Pot)', min_value=2.80, max_value=4.70, value=3.50)
Pot_slider = st.slider('Adjust Potential (Pot)', 2.80, 4.70, value=Pot)

Sn_percent = st.number_input('Sn %', min_value=0.0, max_value=1.0, value=0.5)
Sn_percent_slider = st.slider('Adjust Sn %', 0.0, 1.0, value=Sn_percent)

pH = st.number_input('pH', min_value=8.02, max_value=14.05, value=9.0)
pH_slider = st.slider('Adjust pH', 8.02, 14.05, value=pH)

# Synchronize slider and number input values
cDen = cDen_slider if cDen != cDen_slider else cDen
Pot = Pot_slider if Pot != Pot_slider else Pot
Sn_percent = Sn_percent_slider if Sn_percent != Sn_percent_slider else Sn_percent
pH = pH_slider if pH != pH_slider else pH

if st.button('Calculate'):
    results = calculate_outputs(Sn_percent, Pot, cDen, pH)
    st.write('## Results')
    st.write(results) 