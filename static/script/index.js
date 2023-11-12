
API_URL = "http://localhost:5000"
RED = "bg-red-500"
GREEN = "bg-green-500"

function predict_sentiment(text, target){
  if (text === ""){
    target.innerText = "empty"
    return
  }
  fetch(API_URL + `/nlp/sentiment/predict?text=${text}`, {
    method: "GET"
  }).then(response => response.json())
  .then(data => {
    let result = "breaking the api"
    if (data !== undefined){
       result = data["prediction"]
    }
    target.innerText = result
  })
}

function predict_classification(text, target){
  if (text === ""){
    target.innerText = "unknown"
    return
  }
  fetch(API_URL + `/nlp/classification/predict?text=${text}`, {
    method: "GET"
  }).then(response => response.json())
  .then(data => {
    let result = "breaking the api"
    if (data !== undefined){
       result = data["prediction"]
    }
    target.innerText = result
  })
}

function predict_similarity(text1, text2, target){
  if (text1 === "" || text2 === ""){
    target.innerText = "unknown"
    return
  }
  fetch(API_URL + `/nlp/similarity/predict?text1=${text1}&text2=${text2}`, {
    method: "GET"
  }).then(response => response.json())
  .then(data => {
    let result = "breaking the api"
    if (data !== undefined){
       result = data["prediction"]
    }
    target.innerText = result
    if (result === "unsimilar"){
      target.classList.remove(GREEN)
      target.classList.add(RED)
    } else {
      target.classList.remove(RED)
      target.classList.add(GREEN)
    }
  })
}

document.getElementById("text1").addEventListener("input", (e) => {
  e.preventDefault()
  const text = e.target.value
  predict_sentiment(text, document.getElementById("sentiment1"))
  predict_classification(text, document.getElementById("classification1"))
  predict_similarity(text, document.getElementById("text2").value, document.getElementById("similarity"))
})

document.getElementById("text2").addEventListener("input", (e) => {
  e.preventDefault()
  const text = e.target.value
  predict_sentiment(text, document.getElementById("sentiment2"))
  predict_classification(text, document.getElementById("classification2"))
  predict_similarity(document.getElementById("text1").value, text, document.getElementById("similarity"))
})

