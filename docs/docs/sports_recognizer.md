---
layout: page
title: Sports Equipment Recognizer
---



<input id="photo" type="file" accept="image/*">
<div id="results"></div>

<script>
    async function uploadImage(file) {
        const reader = new FileReader();

        reader.onloadend = async () => {
            const base64Image = reader.result;

            // Display the uploaded image in the results section
            document.getElementById("results").innerHTML = `
                <p>Uploaded Image:</p>
                <img id="uploadedImage" src="${base64Image}" width="300" alt="Uploaded Image">
            `;

            const payload = { data: [base64Image] };

            try {
                const response = await fetch("https://ahmedtanvir47-sport-recognizer.hf.space/run/predict", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const json = await response.json();
                console.log("Full API Response:", json);

                const label = json.data[0].label || "No label found"; // Adjust based on response structure
                document.getElementById("results").innerHTML += `<p>Prediction: ${label}</p>`;

            } catch (error) {
                console.error("Error:", error);
                document.getElementById("results").innerHTML += `<p style="color:red;">An error occurred: ${error.message}</p>`;
            }
        };

        reader.readAsDataURL(file); // Convert file to base64
    }

    document.getElementById("photo").addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            uploadImage(file);
        }
    });
</script>
