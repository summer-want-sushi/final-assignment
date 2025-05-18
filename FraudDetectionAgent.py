import pandas as pd
from sklearn.ensemble import IsolationForest
from smolagents import CodeAgent


class FraudDetectionAgent:
    pass


CodeAgent = FraudDetectionAgent()


class TransactionModel:
    def __init__(self, transaction_id: int, amount: float, timestamp: str, location_lat: float, location_long: float):
        self.transaction_id = transaction_id
        self.amount = amount
        self.timestamp = timestamp
        self.location_lat = location_lat
        self.location_long = location_long

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "location_lat": self.location_lat,
            "location_long": self.location_long
        }


class FraudResult:
    def __init__(self, transaction_id: int, amount: float, timestamp: str, anomaly_score: int):
        self.transaction_id = transaction_id
        self.amount = amount
        self.timestamp = timestamp
        self.anomaly_score = anomaly_score

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "amount": self.amount,
            "timestamp": self.timestamp,
            "anomaly_score": self.anomaly_score
        }


class FraudDetectionAgent(FraudDetectionAgent):
    def __init__(self, data_path: str):
        super().__init__()
        self.data_path = data_path
        self.df = None
        self.X = None
        self.model = IsolationForest(contamination=0.01, random_state=42)

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        print(f"Loaded {len(self.df)} transactions.")

    def preprocess(self):
        self.df['transaction_hour'] = pd.to_datetime(self.df['timestamp']).dt.hour
        features = ['amount', 'transaction_hour', 'location_lat', 'location_long']
        self.df = self.df.dropna(subset=features)
        self.X = self.df[features]

    def detect_fraud(self):
        self.df['anomaly_score'] = self.model.fit_predict(self.X)
        frauds = self.df[self.df['anomaly_score'] == -1]
        print(f"Detected {len(frauds)} potential fraudulent transactions.")
        return [
            FraudResult(
                row['transaction_id'],
                row['amount'],
                row['timestamp'],
                row['anomaly_score']
            ).to_dict()
            for _, row in frauds.iterrows()
        ]

    def run(self):
        self.load_data()
        self.preprocess()
        return self.detect_fraud()


if __name__ == "__main__":
    agent = FraudDetectionAgent(data_path="transactions.csv")
    agent.run()
    print("\nFraud detection completed.")

