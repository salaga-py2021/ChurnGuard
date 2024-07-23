# ChurnGuard

<a name="readme-top"></a>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/salaga-py2021/ChurnGuard">
    <img src="images/churnguard.jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">ChurnGuard</h3>

  <p align="center">
    Streamlined to help your business lock in customer loyalty
    <br />
    <a href="https://github.com/salaga-py2021/ChurnGuard"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://churnguard-li84cj7r3jbm5cxh9u4djz.streamlit.app/">View Demo</a>
  </p>
</div>


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#team-contact">Team Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
<div style="text-align:center">
<a href="https://github.com/salaga-py2021/ChurnGuard">
  <img src="images/demo.png" alt="demo" width=80% height=90%>
</a>
</div>


ChurnGuard is designed to help banks lock in customer loyalty. The platform offers insights into a bank’s customer retention & the factors behind their churning. Then, given a customer being considered by bank employee & some basic information about the customer, we can predict the probability of churn and classify whether the customer is going to churn in the near future.

ChurnGuard was a project made by college students as part of a FinHackathon at UTDallas. To learn more about the winners, please <a href="#team-contact">look at our team</a>. To learn more about the hackathon you can visit <a href="https://jindal.utdallas.edu/news/finhack-2024-challenges-and-encourages-ut-dallas-students/">UTDallas Jindal News Center</a>

Product Offerings:
* Historical customer analysis 
* Customer churn prediction for bank customers
* LLM generated insights for result interpretation and reccomendations to retain customers

ChurnGuard predicts data using popular ensemble models (logistic regression, random forest, ADA boost, XG boost). The model results for each of ensemble models are below. 

<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>Accuracy</th>
      <th>F1</th>
      <th>AUC</th>
      <th>Specificity</th>
      <th>Sensitivity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Random Forest</td>
      <td>83.14%</td>
      <td>59.23%</td>
      <td>0.7541</td>
      <td>88.08%</td>
      <td>62.74%</td>
    </tr>
    <tr>
      <td>ADA Boost</td>
      <td>73.14%</td>
      <td>45.12%</td>
      <td>0.6686</td>
      <td>77.16%</td>
      <td>56.55%</td>
    </tr>
    <tr>
      <td>XG Boost</td>
      <td>81.44%</td>
      <td>56.19%</td>
      <td>0.7368</td>
      <td>86.41%</td>
      <td>60.97%</td>
    </tr>
    <tr>
      <td>Logistic Regression</td>
      <td>59.22%</td>
      <td>37.52%</td>
      <td>0.6055</td>
      <td>58.36%</td>
      <td>62.74%</td>
    </tr>
  </tbody>
</table>


<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

The project was built with the following libraries.

* [![Conda][Conda]][Conda_url]
* [![Numpy][Numpy]][Numpy_url]
* [![Pandas][Pandas]][Pandas_url]
* [![Streamlit][Streamlit]][Streamlit_url]
* [![Scikit Learn][Scikit Learn]][Scikit_url]
* [![ChatGPT][ChatGPT]][ChatGPT_url]


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

Please install the libraries listed in the requirements.txt file 
* npm
  ```sh
  conda install <library name>
  ```

### Installation


1. Get an  API Key at [https://openai.com/](https://openai.com/) for ChatGPT
2. Clone the repo
   ```sh
   git clone https://github.com/harshak913/ChurnGuard
   ```
3. Install Python Packages from requirements.txt
   ```sh
   conda install <library name>
   ```
4. Enter your API as an environment variable called `OPENAI_API_KEY`
   ```python
   openai_api_key = 'ENTER YOUR API'
   ```
5. Run the streamlit library
   ```sh
   streamlit run test_dashboard.py
   ```



<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

To make a prediction, use the website to enter the values for the customer in the text input section. Run the program by pressing the submit button to generate new prediciton for customer. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Improve UI for OpenAI responses
- [ ] Add chat bot feature to allow managers to ask questions about model interpretation
- [X] Finetune underlying models for predictions using a wider hyperparameter space
- [X] Host website on live webserver 
- [ ] Add data about bank to LLM to create better reccommendations to managers

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Team Contact

- Harsha Gurram - [LinkedIn](https://www.linkedin.com/in/harsha-gurram/)
- Harsha Kolachina - [LinkedIn](https://www.linkedin.com/in/harshakolachina/)
- Mihir Padsumbiya  - [LinkedIn](https://www.linkedin.com/in/mihir-padsumbiya/)
- Viswa Kotra - [LinkedIn](https://www.linkedin.com/in/viswa-kotra/)



Project Link: [https://github.com/salaga-py2021/ChurnGuard](https://github.com/salaga-py2021/ChurnGuard)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Kaggle](https://www.kaggle.com/)
* [Event Organizer: FinHack UTDallas](https://utdfinhack.org/)
* [Sponsor: QuantConnect](https://www.quantconnect.com/)


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: data/demo.png
[Conda]: https://img.shields.io/badge/conda-342B029.svg?&style=for-the-badge&logo=anaconda&logoColor=white
[Conda_url]: https://conda.io/projects/conda/en/latest/user-guide/install/index.html
[Numpy]: https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white
[Numpy_url]: https://numpy.org/
[Pandas]: https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white
[Pandas_url]: https://pandas.pydata.org/
[Streamlit]: https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white
[Streamlit_url]: https://streamlit.io/
[Scikit Learn]: https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[Scikit_url]: https://scikit-learn.org/stable/
[ChatGPT]: https://img.shields.io/badge/ChatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white
[ChatGPT_url]: https://openai.com/
