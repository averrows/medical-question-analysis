
API_URL = "http://localhost:5000"

function predict_sentiment(text){
  if (text === ""){
    document.getElementById("sentiment").innerText = "empty"
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
    document.getElementById("sentiment").innerText = result
  })
}

document.getElementById("text").addEventListener("input", (e) => {
  e.preventDefault()
  const text = e.target.value
  predict_sentiment(text)
})