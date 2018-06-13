$(function(){
  var req;
  d = new Date();
  var url = "http://192.168.1.107/cgi-bin/api.cgi?cmd=Snap&channel=0&user=root&password=123456" + '&'+ d.getTime();
  $('#video-stream').css('background-image', 'url(' + url + ')');

  //send request to IP Camera
  var ctrlUrl="http://192.168.1.107/cgi-bin/api.cgi?user=root&password=123456&cmd=PtzCtrl&token=141263da251ab9e";
  var json;
  $('#btn-up').on('mousedown', function(e) {
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Up", "speed": 32}}];
    makeRequest();
  }).on('mouseup', function() {
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    makeRequest();
  });
  $('#btn-down').on('mousedown', function(e) {
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Down", "speed": 32}}];
    makeRequest();
  }).on('mouseup', function() {
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    makeRequest();
  });
  $('#btn-left').on('mousedown', function(e) {
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Left", "speed": 32}}];
    makeRequest();
  }).on('mouseup', function() {
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    makeRequest();
  });
  $('#btn-right').on('mousedown', function(e) {
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Right", "speed": 32}}];
    makeRequest();
  }).on('mouseup', function() {
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    makeRequest();
  });
  $('#btn-test').click(function(e){
    e.preventDefault();
    json = [{"cmd":"PtzCtrl","action":0,"param":{"channel":0,"op":"ToPos","speed":32,"id":1}}];
    makeRequest();
  });

  function makeRequest(){
    //create request for shopify
    req = new XMLHttpRequest();

    // req.onreadystatechange=function(e) {
    //   if(req.readyState==4) {
    //     var showErrorTab=false;
    //     if(req.status==200) {
    //       console.log("response:"+req.responseText);
    //     } else {
    //       console.log("Error calling PtzCtrl");
    //     }
    //   }
    // }

    req.open("POST",ctrlUrl);
    req.setRequestHeader("Accept","application/json");
    req.send(JSON.stringify(json));
  }




  // setInterval(function(){
  //   $('#video-stream').css('background-image', '');
  // },2000)
});
