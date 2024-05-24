import time
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


class TestLoginPage:
    def test_valid_login(self, driver):
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
        assert "Upload Your Files" in driver.page_source  # Adjust the assertion as needed

        print("Test passed: Login successful")

        # time.sleep(2)

    def test_page_switching(self, driver):
        driver.get("http://localhost:8501")  # Change to your Streamlit app URL
        driver.maximize_window()

        # Allow some time for the page to load
        time.sleep(3)

        ### Login part
        # Find and fill the login form elements
        username_input = driver.find_element(By.XPATH, "//input[@aria-label='Username']")  # Adjust if needed
        password_input = driver.find_element(By.XPATH, "//input[@aria-label='Password']")
        #password_input = driver.find_element(By.NAME, "Password")  # Adjust if needed

        username_input.send_keys("azer")
        password_input.send_keys("Ach123456*")
        
        # Submit the form
        login_button = driver.find_element(By.XPATH, '//button[@kind="secondary"]')  # Adjust the locator as needed
        login_button.click()
        ###

        # Wait for the sidebar to be present
        sidebar = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//ul[@data-testid ='stSidebarNavItems']"))
        )

        # Find the list item by its text and click on it
        page_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//li/div/a[@href='http://localhost:8501/Twin_Screw_ANN']"))
        )

        page_link.click()
        print("Clicked on 'Page 2'")

        # Allow some time for the response
        time.sleep(3)

        # Check the result - adjust the condition to match your app's behavior
        assert "Twin Screw Page" in driver.page_source  # Adjust the assertion as needed

        print("Test passed: Switching between pages works")
    