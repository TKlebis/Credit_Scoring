import pandas as pd
import streamlit as st
from io import BytesIO
from pycaret.classification import setup, load_model, predict_model
from sklearn.preprocessing import LabelEncoder


def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')


def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def main():
    st.set_page_config(
        page_title='PyCaret',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    st.write("## Escorando o modelo gerado no PyCaret")
    st.markdown("---")

    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type=['csv', 'ftr'])

    if data_file_1 is not None:
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)

        df_encoded = df_credit.copy()

        label_encoder = LabelEncoder()
        categorical_columns = ['data_ref', 'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia']
        for column in categorical_columns:
            df_encoded[column] = label_encoder.fit_transform(df_encoded[column])

        reg = setup(data=df_encoded, target='mau', session_id=123, log_experiment=True, log_plots=True)

        model_saved = load_model('modelo_treinado')
        predict = predict_model(model_saved, data=df_encoded)

        df_xlsx = to_excel(predict)
        st.download_button(label='📥 Download', data=df_xlsx, file_name='predict.xlsx')

if __name__ == '__main__':
    main()






