let normalizationData = null;
let allow = 1
let fligthSatisfactionValue = document.querySelector(".fligth_satisfaction_value_container")
let loadingImage = document.querySelector(".loading-image-container")

async function predict(inputFeatures,path, key) {

    const session = await ort.InferenceSession.create(path);

    const input = new Float32Array(inputFeatures);
    const tensor = new ort.Tensor('float32', input, [1, inputFeatures.length]);

    const feeds = {};
    feeds[key] = tensor

    const result = await session.run(feeds);

    return result;
}

async function loadNormalizationInfo() {
  if (normalizationData) {
    return normalizationData;
  }

  const response = await fetch('./models/normalization_info.json');
  normalizationData = await response.json(); 
  return normalizationData;
}

async function normalizeInputs(inputs) {
  const normalizationJsonInfo = await loadNormalizationInfo(); 

  let result = { status: true, message: '', data: [] };

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];

    if (input.value == null || input.value == undefined || input.value === '') {
      result.status = false;
      result.message = 'The ' + input.getAttribute('placeholder') + ' is empty.';
      break;
    } 

    if (input.getAttribute("min") !== null && input.getAttribute("max")){

      let message = `The ${input.getAttribute('placeholder')} must be in range of ${input.getAttribute("min")} and ${input.getAttribute("max")}.`;


      if (input.value > input.getAttribute("max")){
        result.status = false;
        result.message = message;
        break;
      }
      else if (input.value < input.getAttribute("min")){
        result.status = false;
        result.message = message;
        break;
      }

    }

    let input_min_value = normalizationJsonInfo.min_values[input.getAttribute("data-key")]
    let input_max_value = normalizationJsonInfo.max_values[input.getAttribute("data-key")]

    let normalizeInput = (input.value - input_min_value) / (input_max_value - input_min_value);

   // normalizeInput = parseFloat(normalizeInput.toFixed(6));

    result.data.push(normalizeInput)
  }

  return result;
}


async function deNormalizeValue(normalizedValue, target) {

    const normalizationJsonInfo = await loadNormalizationInfo();
  
    let value_min_value = normalizationJsonInfo.min_values[target]
    let value_max_value = normalizationJsonInfo.max_values[target]
    
    value = (normalizedValue * (value_max_value - value_min_value)) + value_min_value;

    return Math.round(value)
  }

document.querySelector('.calculate').addEventListener('click', async () => {

  if (allow == 1){
    
  allow = 0  
  fligthSatisfactionValue.classList.add("hidden")
  loadingImage.classList.remove("hidden")

  const values = document.querySelectorAll('.pred-input');

  const normalizeValues = await normalizeInputs(values); 

  if(!normalizeValues.status){
    allow = 1 
    loadingImage.classList.add("hidden")
    return window.alert(normalizeValues.message);
  }

  const algorithm = document.querySelector(".algorithm-input").value
  let prediction = 0
  
  if (algorithm == "neural_network"){
    prediction = await predict(normalizeValues.data, "./models/deep_learning_model.onnx", "input")
    prediction = prediction.output.cpuData[0]
  }
  else {
    prediction = await predict(normalizeValues.data, "./models/rf_model.onnx", "float_input")
    prediction = prediction.output_label.cpuData[0]
  }


  let finalPrediction = prediction > 0.5 ? "satisfied" : "neutral or dissatisfied"

  loadingImage.classList.add("hidden")
  fligthSatisfactionValue.classList.remove("hidden")
  document.querySelector("#fligth_satisfaction_value_value").innerHTML = finalPrediction
  allow = 1
  
}
});
