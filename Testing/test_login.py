from selenium import webdriver
from selenium.webdriver.common.by import By
import time

# Initialize WebDriver
driver = webdriver.Edge()

try:
    # Open the Streamlit app
    driver.get("http://localhost:8501")  # Change to your Streamlit app URL
    driver.maximize_window()

    # Allow some time for the page to load
    time.sleep(3)

    # Find and fill the login form elements
    username_input = driver.find_element(By.XPATH, "//input[@aria-label='Username']")  # Adjust if needed
    password_input = driver.find_element(By.XPATH, "//input[@aria-label='Password']")
    #password_input = driver.find_element(By.NAME, "Password")  # Adjust if needed

    username_input.send_keys("azer")
    password_input.send_keys("Ach123456*")
    
    # Submit the form
    login_button = driver.find_element(By.XPATH, '//button[@kind="secondary"]')  # Adjust the locator as needed
    login_button.click()

    # Allow some time for the response
    time.sleep(3)

    # Check the result - adjust the condition to match your app's behavior
    assert "Login successful!" in driver.page_source  # Adjust the assertion as needed

    print("Test passed: Login successful")

except Exception as e:
    print(f"Test failed: {e}")

finally:
    # Close the browser
    driver.quit()