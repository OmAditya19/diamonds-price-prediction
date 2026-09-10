from src.predict import predict_price


price = predict_price(
    carat=1.0,
    cut="Ideal",
    color="G",
    clarity="VS1",
    depth=61.5,
    table=57.0,
)

print(f"Predicted price: ${price:,.2f}")
