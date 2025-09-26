import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import ssl
import certifi

# Fix SSL issues
ssl._create_default_https_context = ssl._create_unverified_context
import os

os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

print("Démarrage du programme d'analyse financière...")


# -----------------------------
# 1️⃣ CRÉATION DE DONNÉES RÉALISTES (AMÉLIORÉE)
# -----------------------------
def create_realistic_sample_data():
    """Crée des données boursières réalistes avec tendances et volatilité différentes"""
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq='D')
    np.random.seed(42)

    # Tendances différentes pour chaque action
    trends = {
        'AAPL': 0.0005,  # Légère tendance haussière
        'MSFT': 0.0003,  # Tendence modérée
        'GOOGL': 0.0001,  # Tendence faible
        'TSLA': 0.0010,  # Forte volatilité
        'AMZN': 0.0004  # Tendence stable
    }

    volatilities = {
        'AAPL': 0.02,
        'MSFT': 0.015,
        'GOOGL': 0.012,
        'TSLA': 0.04,
        'AMZN': 0.018
    }

    data = {}
    prices_init = {'AAPL': 150, 'MSFT': 300, 'GOOGL': 2500, 'TSLA': 200, 'AMZN': 130}

    for ticker in ['AAPL', 'MSFT', 'GOOGL']:  # On garde 3 actions pour la démo
        # Génération de prix plus réaliste avec tendance + bruit
        returns = np.random.normal(trends[ticker], volatilities[ticker], len(dates))
        prices = prices_init[ticker] * np.exp(np.cumsum(returns))
        data[ticker] = prices

    return pd.DataFrame(data, index=dates)


# Utiliser les données de démonstration réalistes
print("Création de données boursières réalistes...")
data = create_realistic_sample_data()
tickers = data.columns.tolist()

print("Données créées :")
print(data.head())
print(f"\nPériode: {data.index[0].date()} to {data.index[-1].date()}")
print(f"Nombre de jours: {len(data)}")

# -----------------------------
# 2️⃣ ANALYSE STATISTIQUE (CORRIGÉE)
# -----------------------------
returns = data.pct_change().dropna()

mean_returns = returns.mean()
volatility = returns.std()
correlation = returns.corr()

print("\n" + "=" * 50)
print("ANALYSE STATISTIQUE RÉALISTE")
print("=" * 50)

print("\n📈 Rendements moyens quotidiens:")
for ticker in tickers:
    print(f"  {ticker}: {mean_returns[ticker]:.4%}")

print("\n📊 Volatilité quotidienne:")
for ticker in tickers:
    print(f"  {ticker}: {volatility[ticker]:.4%}")

print("\n🔄 Matrice de corrélation:")
print(correlation.round(3))

# -----------------------------
# 3️⃣ VISUALISATION AMÉLIORÉE
# -----------------------------
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Graphique 1: Prix historiques
axes[0, 0].set_title('Évolution des Prix', fontweight='bold')
for ticker in tickers:
    axes[0, 0].plot(data.index, data[ticker], label=ticker, linewidth=2)
axes[0, 0].set_ylabel('Prix ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Graphique 2: Rendements cumulés
cumulative_returns = (1 + returns).cumprod()
axes[0, 1].set_title('Rendements Cumulés', fontweight='bold')
for ticker in tickers:
    axes[0, 1].plot(cumulative_returns.index, cumulative_returns[ticker], label=ticker, linewidth=2)
axes[0, 1].set_ylabel('Rendement Cumulé (1 = 100%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Graphique 3: Matrice de corrélation
im = axes[1, 0].imshow(correlation, cmap='RdYlBu', aspect='auto', vmin=-1, vmax=1)
axes[1, 0].set_title('Matrice de Corrélation', fontweight='bold')
plt.colorbar(im, ax=axes[1, 0])
axes[1, 0].set_xticks(range(len(tickers)))
axes[1, 0].set_yticks(range(len(tickers)))
axes[1, 0].set_xticklabels(tickers)
axes[1, 0].set_yticklabels(tickers)

# Graphique 4: Volatilité
axes[1, 1].bar(tickers, volatility.values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
axes[1, 1].set_title('Volatilité des Actions', fontweight='bold')
axes[1, 1].set_ylabel('Volatilité Quotidienne')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# -----------------------------
# 4️⃣ OPTIMISATION DU PORTEFEUILLE (AMÉLIORÉE)
# -----------------------------
def optimize_portfolio(returns, num_portfolios=15000):
    """Optimisation robuste du portefeuille"""
    results = np.zeros((4, num_portfolios))  # +1 pour les poids
    weights_record = []

    mean_returns_annual = returns.mean() * 252
    cov_matrix_annual = returns.cov() * 252

    for i in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(len(tickers)))  # Meilleure distribution
        port_return = np.dot(weights, mean_returns_annual)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annual, weights)))
        sharpe_ratio = port_return / port_volatility if port_volatility > 0 else 0

        results[0, i] = port_return
        results[1, i] = port_volatility
        results[2, i] = sharpe_ratio
        weights_record.append(weights)

    # Meilleur portefeuille par ratio de Sharpe
    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    return optimal_weights, results[:, max_sharpe_idx]


print("\n" + "=" * 50)
print("OPTIMISATION DU PORTEFEUILLE")
print("=" * 50)

optimal_weights, optimal_results = optimize_portfolio(returns)

print("\n🎯 PORTEFEUILLE OPTIMAL (Max Ratio de Sharpe):")
print("-" * 45)
total_weight = 0
for ticker, weight in zip(tickers, optimal_weights):
    if weight > 0.01:  # Afficher seulement les poids significatifs
        print(f"  {ticker}: {weight:.2%}")
        total_weight += weight

print(f"  Total: {total_weight:.2%}")

print(f"\n📊 Performance annuelle attendue:")
print(f"  ▸ Rendement: {optimal_results[0]:.2%}")
print(f"  ▸ Volatilité: {optimal_results[1]:.2%}")
print(f"  ▸ Ratio de Sharpe: {optimal_results[2]:.2f}")

# -----------------------------
# 5️⃣ PRÉDICTIONS IA (CORRIGÉE)
# -----------------------------
print("\n" + "=" * 50)
print("PRÉDICTIONS INTELLIGENCE ARTIFICIELLE")
print("=" * 50)

future_days = 30
predictions_dict = {}

print("Entraînement des modèles de prédiction...")

for ticker in tickers:
    try:
        # Préparer les données
        X = np.arange(len(data)).reshape(-1, 1)
        y = data[ticker].values

        # Normalisation
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Modèle plus robuste
        model = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=3000,
            random_state=42,
            alpha=0.001,
            learning_rate_init=0.001
        )

        model.fit(X_scaled, y_scaled)

        # Prédictions
        future_X = np.arange(len(data), len(data) + future_days).reshape(-1, 1)
        future_X_scaled = scaler_X.transform(future_X)
        pred_scaled = model.predict(future_X_scaled)
        predictions = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        predictions_dict[ticker] = predictions
        last_prediction = predictions[-1]  # ✅ CORRECTION ICI

        print(f"✅ {ticker}: Prédiction à 30j = ${last_prediction:.2f}")

    except Exception as e:
        print(f"❌ {ticker}: Erreur - {str(e)[:100]}...")
        # Valeur par défaut réaliste
        predictions_dict[ticker] = np.full(future_days, data[ticker].iloc[-1])

# -----------------------------
# 6️⃣ VISUALISATION DES PRÉDICTIONS
# -----------------------------
print("\n🔄 Génération des graphiques de prédiction...")

plt.figure(figsize=(15, 10))
for i, ticker in enumerate(tickers, 1):
    plt.subplot(2, 2, i)

    # Derniers 60 jours historiques
    historical_data = data[ticker].iloc[-60:]
    plt.plot(historical_data.index, historical_data.values, 'b-',
             label='Historique', linewidth=2, alpha=0.8)

    # Prédictions
    future_dates = pd.date_range(start=data.index[-1], periods=future_days + 1, freq='D')[1:]
    plt.plot(future_dates, predictions_dict[ticker], 'r--',
             label='Prédiction (30j)', linewidth=2)

    # Connexion
    plt.plot([data.index[-1], future_dates[0]],
             [data[ticker].iloc[-1], predictions_dict[ticker][0]],
             'r--', alpha=0.5)

    plt.title(f'{ticker} - Prévision des Prix', fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Prix ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Infobox
    current_price = data[ticker].iloc[-1]
    predicted_price = predictions_dict[ticker][-1]
    change_pct = ((predicted_price - current_price) / current_price) * 100

    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8)
    plt.text(0.02, 0.98,
             f'Prix actuel: ${current_price:.2f}\n'
             f'Prévision: ${predicted_price:.2f}\n'
             f'Variation: {change_pct:+.1f}%',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=bbox_props, fontsize=9)

plt.tight_layout()
plt.show()

# -----------------------------
# 7️⃣ RAPPORT FINAL DÉTAILLÉ
# -----------------------------
print("\n" + "=" * 60)
print("📊 RAPPORT FINAL D'ANALYSE FINANCIÈRE")
print("=" * 60)

print(f"\n📈 PERFORMANCE SUR LA PÉRIODE ({len(data)} jours):")
print("-" * 55)

for ticker in tickers:
    initial_price = data[ticker].iloc[0]
    final_price = data[ticker].iloc[-1]
    total_return = ((final_price / initial_price) - 1) * 100
    predicted_price = predictions_dict[ticker][-1]
    future_return = ((predicted_price / final_price) - 1) * 100

    print(f"\n{ticker}:")
    print(f"  ▸ Prix initial: ${initial_price:.2f}")
    print(f"  ▸ Prix final: ${final_price:.2f}")
    print(f"  ▸ Rendement période: {total_return:+.2f}%")
    print(f"  ▸ Prévision 30j: ${predicted_price:.2f}")
    print(f"  ▸ Variation attendue: {future_return:+.2f}%")
    print(f"  ▸ Volatilité: {volatility[ticker]:.4f}")

print(f"\n💡 ANALYSE DU PORTEFEUILLE OPTIMAL:")
print("-" * 55)
print("Répartition recommandée:")
for ticker, weight in zip(tickers, optimal_weights):
    if weight > 0.05:
        amount = weight * 10000  # Pour un portefeuille de $10,000
        print(f"  ▸ {ticker}: {weight:.1%} (${amount:.0f})")

print(f"\n📋 CARACTÉRISTIQUES ATTENDUES:")
print(f"  ▸ Rendement annuel: {optimal_results[0]:.1%}")
print(f"  ▸ Volatilité annuelle: {optimal_results[1]:.1%}")
print(f"  ▸ Ratio de Sharpe: {optimal_results[2]:.2f}")

print("\n" + "=" * 60)
print("🎉 ANALYSE TERMINÉE AVEC SUCCÈS!")
print("=" * 60)