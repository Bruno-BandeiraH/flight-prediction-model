"""
Feature Engineering para transformação de dados.
"""
import pandas as pd


def gerar_features(data: dict, features_esperadas: list) -> pd.DataFrame:
    """
    Gera features a partir dos dados brutos do voo.
    
    Args:
        data: Dicionário com dados do voo
        features_esperadas: Lista com ordem esperada das features
        
    Returns:
        DataFrame pandas com features processadas
    """
    df = pd.DataFrame([data])

    # Hora fracionada (ex: 14.5 = 14:30)
    df["hora_prevista_frac"] = (
        df["hora_prevista"].dt.hour +
        df["hora_prevista"].dt.minute / 60
    )

    # Faixa horária (exemplo simples)
    def faixa_horaria(h):
        if 6 <= h < 12:
            return "manha"
        elif 12 <= h < 18:
            return "tarde"
        elif 18 <= h < 24:
            return "noite"
        else:
            return "madrugada"

    df["faixa_horaria"] = df["hora_prevista"].dt.hour.apply(faixa_horaria)

    # Fim de semana
    df["eh_fim_de_semana"] = df["hora_prevista"].dt.weekday >= 5

    # Mês do voo
    df["mes"] = df["hora_prevista"].dt.month

    # Remover coluna original
    df = df.drop(columns=["hora_prevista"])

    # Garantir ordem e presença das features
    df = df[features_esperadas]

    return df

