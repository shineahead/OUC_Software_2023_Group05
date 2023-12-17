window.onload = function () {
    document.getElementById("box2").onclick = function () {
        window.scrollTo({
            top: 0,
            behavior: 'smooth' // 平滑滚动
        });
    }
    document.getElementById("box1").onmouseenter = function () {
        document.getElementById("hoverQR").style.display = "block"
    }
    document.getElementById("box1").onmouseleave = function () {
        document.getElementById("hoverQR").style.display = "none"
    }
}