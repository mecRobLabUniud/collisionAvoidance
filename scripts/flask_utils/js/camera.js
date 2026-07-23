// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────
const socket = io();

// ─────────────────────────────────────────────────────────────────────────────
// Streaming updates
// ─────────────────────────────────────────────────────────────────────────────
function update_stream1() {
    socket.on('update_stream1', function (data) {
        document.getElementById('image1').src = data.frame;
    });
}

function update_stream2() {
    socket.on('update_stream2', function (data) {
        document.getElementById('image2').src = data.frame;
    });
}

function update_stream3() {
    socket.on('update_stream3', function (data) {
        document.getElementById('image3').src = data.frame;
    });
}

function update_stream4() {
    socket.on('update_stream4', function (data) {
        document.getElementById('image4').src = data.frame;
    });
}

setTimeout(update_stream1, 1 / 30 * 1000);
setTimeout(update_stream2, 1 / 30 * 1000);
setTimeout(update_stream3, 1 / 30 * 1000);
setTimeout(update_stream4, 1 / 30 * 1000);



