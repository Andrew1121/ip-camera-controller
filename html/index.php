<!DOCTYPE html>
<html lang="en" dir="ltr">
  <?php
    include '../php/connectdb.php';
  ?>
  <head>
    <meta charset="utf-8">
    <title></title>
    <link rel="stylesheet" type="text/css" href="../css/all.css">
  </head>
  <body>
    <div class="loading-overlay">
      <img src="../images/loading.gif" alt=""/>
    </div>
    <div class="stream-content">
      <div id="video-stream" class="component">
        <!-- <video src="rtsp://andrew1121:panaga1121@192.168.1.107:554//h264Preview_01_main">
        </video> -->
      </div>
      <div class="control-panel component">
        <div class="btn-wrapper">
          <a href="#" id="btn-up" class="control-btn"></a>
          <a href="#" id="btn-down" class="control-btn"></a>
          <a href="#" id="btn-left" class="control-btn"></a>
          <a href="#" id="btn-right" class="control-btn"></a>
        </div>
        <div class="current-state">
          <span>Current Pan Value:</span>
          <p id="pan">N/A</p>
          <span>Current Tilt Value:</span>
          <p id="tilt">N/A</p>
        </div>
        <div class="zoom-panel">
          <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAQoSURBVGhD1drZy29THMfxxzwlU4rMMxeGCHdKUQgdEhkukJQyRRkjiogbQyFjx+U5Nw6SP0DmIdMVmWdljCLDeb91Vn3bffd59tprn+dsn3rVaZ/fXnut/dt7Tb9noSE7Yxmuw/14HI/iXlyNk7AdZpldcRPexb8D/I1XcDl2wHqPDXgMfyCr8BC/4m5siyXPBrgMViKrnP7Ex3gdb+Iz/IXss/oGZ2LJsjWeQVaZ93A7jsIm6GYzHIt78BGyMh5Cdu6k2RHe4e7FvePHoSZ+q2fhA3TLew5bYp1kG7yGeMHfcT6s1NhsCnu47mO3Chth0ljRpxEv9AWOwFQ5AT8hXuM+TJorEC/wOXbD1DkEsTH/4BRMErvY2Dv9hiMxJLfgzjWu9MCAnIr4mH2KrdAcx4lSqC7A0MS7+44HBuY2xGtej6b4bcTB7g3UvNhjG2KP9SXKud9jC4zOjSiFqbaLHdsQcxHitc/G6DjAlYLe90BlWhrioPgtyvl2x6OyE0ohcsSuTUtDzIMo5/+MjVEdp+KlEB2N2rQ25HTEOhyG6jjalgKcAI65G60N2RPlfDmlqY6LolLAhx5IcjN+WAsHtFKGY0P2meJCdLMh4phyFarzBEoBb3kgyR0on2l1CbL8iPIZB9jqxIHwbQ8kWYqGxMdz1MDoGrsU8IkHklwKp/V94mPhTDn7THEGuvG9jI9n9vgtGp/HUoAV2hy1aX3Z90c5XyejOu52xEJc2dWmtSHnItZhX1THLRt3O0ohLk9r09qQ5Sjnu+YfnVdRCnIjoXYl2NIQJ46O5uV898dGx32nUpBqdzlaGnIt4rVdQY6Om2dxUeXAWLPDMbYh2yOOH05YW/YF/stdKAXKOzU0YxvyMOI1R3W73bgDGKfTdsUnYkjcYbl4jWycyOJnYyMcY0bNerP4bsTCvdOjZqKL5HjEFan/PhiTxh3A2JhfMGqA6onfhLPseA036iaPG2nPIl7IceZWtKyn7VC6NylyI8Lyj8Fk3459e7bv+xVcY9f0aJZ1DWLv1Md5Wvn3i9gDzfHFc0IZJ3OFncIDOA17wfVEieftB6cdTyL2aIUVfr5zLONe196YJL4fFphdqPDRKwun7P+jF+Ak0fHCrdLsM9GkjXEH8AZ8h+xiQ7hoOw/x2+vO8/o4/9oHk8WX8Rw8BXuz7KKR87ZH0LdP5jiVnZdxH9pHdvL4LhwKNwr88dPlqSs7t1p9HIfcwcORVbqPvwz4WM4yLyGrdB+3WA/A7LI7FutMuhwGDsTsYq/kC51Vus/XOAizi++UL3RW6T42xl8SZhfX6rWNWYlZxi7W3imrdMZBeLaxi40/AK2NjZ517GLtnbLKR65sZx+7WF/orAF6GU0/1S1l7GL9+5WsEf5xw/8qu2AFXNP4TvhXRuvsTz/WYxYWVgOtr5qV6bpYZQAAAABJRU5ErkJggg==">
          <a href="#" id="btn-zoom1"><span>x1</span></a>
          <a href="#" id="btn-zoom15"><span>x1.5</span></a>
          <a href="#" id="btn-zoom20"><span>x2</span></a>
        </div>
      </div>
    </div>
    <script src="https://code.jquery.com/jquery-2.1.4.min.js"></script>
    <script src="../js/jquery.network-camera.js"></script>
    <script src="../js/all.js"></script>
  </body>
</html>
