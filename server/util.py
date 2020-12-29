# main imports
import json
import pickle
import numpy as np
import os

# Variables globales
__data_columns = None
__model = None
__genders = None
__education = None
__marital_status = None
__dependents = None
__self_employed = None
__property_area = None

# Rutina que nos devolverá la predicción del modelo
def get_model_prediction(ap_income, coap_income, loan_amt, loan_amt_term, credit_hist, 
                         gender, education, married_status, dependents, self_employed, 
                         property_area):
    
    global __model

    param_list = [gender, education, married_status, dependents, self_employed, property_area]
    index_list = []

    for param in param_list:
        try:
            index = __data_columns.index(param.lower())
        except:
            index = -1
        
        index_list.append(index)

    x = np.zeros(len(__data_columns))

    x[0] = ap_income
    x[1] = coap_income
    x[2] = loan_amt
    x[3] = loan_amt_term
    x[4] = credit_hist

    for ind in index_list:
        if ind >= 0:
            x[ind] = 1

    # x is my numpy array
    prediction = __model.predict([x])

    if prediction == 1:
        return "Approved"
    else:
        return "Rejected"

# Cargar archivos en artifacts
def load_saved_artifacts():

    print("Loading saved artifacts...start")

    global __data_columns, __genders, __education, __marital_status
    global __dependents, __self_employed, __property_area, __model

    path = os.path.dirname(os.path.abspath(__file__))
    artifacts = os.path.join(path, "artifacts")

    # Cargamos datos desde el archivo json
    with open(artifacts + "/columns.json", "r") as f:
        __data_columns = json.load(f)["data_columns"]

        # indices donde se encuentran los valores para "Gender"
        __genders = __data_columns[5:7]

        # indices donde se encuetran los valores de "Education"
        __education = __data_columns[7:9]

        # indices donde se encuentran los valores de "Married"
        __marital_status = __data_columns[9:11]

        # indices donde se encuentran los valores de "Dependents"
        __dependents = __data_columns[11:15]

        # indices donde se encuentran los valores de self employed
        __self_employed = __data_columns[15:17]

        # indices donde se encuentras los valores de property area
        __property_area = __data_columns[17:]

    # cargamos modelo guardado en archivo pickle
    with open(artifacts + "final_model.pkl", "rb") as f:
        __model = pickle.load(f)

    print("Loading saved artifacts...done")

# Como utilizamos mucho OHE en el modelo, necesitamos obtener el nombre de las opciones en cada columna

# Para obtener el sexo del aplicante
def get_genders():
    return __genders

# Para obtener educación del aplicante
def get_education():
    return __education

# Para obtener estado de matrimonio del aplicante
def get_marital_status():
    return __marital_status

# Para obtener dependents
def get_dependents():
    return __dependents

# Para obtener self_employed
def get_self_employed():
    return __self_employed

# Para obtener property_area
def get_property_area():
    return __property_area

if __name__ == "__main__":
    load_saved_artifacts()
    print(get_genders())
    print(get_education())
    print(get_marital_status())
    print(get_dependents())
    print(get_self_employed())
    print(get_property_area())
    #print(get_model_prediction(4500, 2000, 115, 360, 1.0, "Male", "Not Graduate", "Married_No", "0", "Self_Employed_No", "Semiurban"))
