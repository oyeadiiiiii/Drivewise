const eventSource = new EventSource("/state_feed");

let prevState = ""; // Variable to store the previous state

function updateDriverName() {
    fetch('/get_driver_name')
    .then(response => response.json())
    .then(data => {
        const driverName = document.getElementById("driverName");
        driverName.textContent = "Driver's Name: " + data.driverName;
    })
    .catch(error => {
        console.error('Error fetching driver name:', error);
    });
}

eventSource.onmessage = function(event) {
    const stateText = document.getElementById("stateText");
    let newState = event.data.trim();

    if (newState !== prevState && newState !== "") {
        stateText.textContent += newState + "\n";
        stateBox.scrollTop = stateBox.scrollHeight;
        prevState = newState;
    }
};

document.addEventListener('DOMContentLoaded', () => {
    const registerButton = document.getElementById('registerButton');

    registerButton.addEventListener('click', () => {
        const driverName = prompt("Please enter your name:", "");

        if (driverName !== null && driverName !== "") {
            fetch('/get_driver_name')
            .then(response => response.json())
            .then(data => {
                if (data.driverName === null || data.driverName !== driverName) {
                    fetch('/register_driver', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ name: driverName })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Driver registered successfully!');
                            updateDriverName();
                        } else {
                            alert('Failed to register driver.');
                        }
                    })
                    .catch(error => {
                        console.error('Error registering driver:', error);
                    });
                } else {
                    alert('Driver already registered.');
                }
            })
            .catch(error => {
                console.error('Error fetching driver name:', error);
            });
        }
    });

    updateDriverName();
});

eventSource.onerror = function(error) {
    console.error("EventSource failed:", error);
    eventSource.close();
};

navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 4096 }, height: { ideal: 2160 } } })
    .then(stream => {
        const videoElement = document.getElementById('videoElement');
        videoElement.srcObject = stream;
    })
    .catch(err => {
        console.error("Error accessing the webcam:", err);
    });
