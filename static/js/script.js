const form = document.getElementById("predictionForm");

const button = document.getElementById("predictButton");

const buttonText = document.getElementById("buttonText");

const loadingText = document.getElementById("loadingText");


form.addEventListener("submit", function () {

    /*
        Prevent multiple clicks while
        the model is processing the request.
    */

    button.disabled = true;

    buttonText.classList.add("hidden");

    loadingText.classList.remove("hidden");

});