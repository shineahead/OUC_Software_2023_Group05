// 设置一个标志位，表示弹窗是否已经显示
var popupDisplayed = false;
var popup = document.getElementById("popup")
// 设置滚动监听器
window.addEventListener("scroll", function () {
    // 获取滚动的垂直距离
    var scrollDistance = window.scrollY;

    // 设置一个滚动距离的阈值，例如 200 像素
    var scrollThreshold = 900;

    // 判断滚动距离是否超过阈值且弹窗未显示
    if (scrollDistance > scrollThreshold && !popupDisplayed) {
        popup.style.display = "block";
        setTimeout(function () {
            popup.style.transform = "translate(-50%, -50%) scale(1)";
        }, 0);
        popupDisplayed = true;
    }
});

document.getElementById("close_popup_button").addEventListener("click", function () {
    popup.style.display = "none";
})

document.getElementById('nav-button').addEventListener("click", function () {
    window.scrollTo({
        top: 960,
        behavior: 'smooth'
    })

})


