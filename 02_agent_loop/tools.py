from langchain_core.tools import tool
import yfinance as yf

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a stock ticker symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., AAPL for Apple Inc.)

    Returns:
        A short string with the current price, or an error message.    
    
    """
    try:
        stock = yf.Ticker(symbol)

        info = stock.info
        if price := info.get("currentPrice") or info.get("regularMarketPrice"):
            return f"The current price of {symbol.upper()} is ${price:.2f}."
        
        hist = stock.history(period="1d")
        if not hist.empty:
            return f"The current price of {symbol.upper()} is ${float(hist['Close'].iloc[-1]):.2f}."
        
        return f"Unable to fetch the current price for {symbol.upper()}."
    except Exception as e:
        return f"Error fetching {symbol.upper()}: {e}"
            

if __name__ == "__main__":
    print(get_stock_price("AAPL"))
    print(get_stock_price("GOOGL"))
    print(get_stock_price("INVALID"))
