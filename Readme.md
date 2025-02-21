# Online Payment Fraud Detection

This project aims to build a machine learning-based system for detecting fraudulent online payment transactions.
It uses various classification algorithms to predict whether a given transaction is legitimate or 
fraudulent based on transaction data such as user behavior, payment details, and other transaction characteristics.

[link to dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)

[Youtube Video](https://www.youtube.com/watch?v=Lb0JbeUjyjs&ab_channel=dayabansagar)

[Connect with me](https://www.linkedin.com/in/dayabansgr/)


## Table of Contents

- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset used for this project is sourced from Kaggle:

[Online Payment Fraud Detection Dataset](https://www.kaggle.com/datasets/jainilcoder/online-payment-fraud-detection)

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Daya-1-sgr/Online_payment_fraud_prediction.git
   cd Online_payment_fraud_prediction
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python3 -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

After installation, you can run the Streamlit application to interact with the fraud detection model:

```bash
streamlit run streamlit_app.py
```

This command will launch a web interface where you can input transaction details and receive predictions on their legitimacy.

Or you can visit :
[Online Payment Fraud Detector](https://onlinepaymentfraudprediction.streamlit.app/)

Please Visit the [Youtube Video](https://www.youtube.com/watch?v=Lb0JbeUjyjs&ab_channel=dayabansagar) for more clear explanation


## Features

- **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
- **Model Training:** Implementing various classification algorithms to identify the best-performing model.
- **Model Evaluation:** Assessing models using metrics like accuracy, precision, recall, and F1-score.
- **Real-time Prediction:** Providing a user-friendly interface for real-time transaction fraud detection.

## Contributing

Contributions are welcome! If you'd like to enhance the project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes relevant tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for providing the dataset.
- The open-source community for continuous support and contributions.

---

*Note: This project is for educational purposes and may require further development for production use.*
