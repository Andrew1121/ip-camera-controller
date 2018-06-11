$(function(){

  // $('#video-stream').networkCamera({
  //   url: 'http://192.168.1.107/cgi-bin/api.cgi?cmd=Snap&channel=0&user=admin&password=a1121@C1123',
  // });
  // $('#video-stream').networkCamera('stream');
  // setInterval(function(){
  //   d = new Date();
  //   var url = "http://192.168.1.107/cgi-bin/api.cgi?cmd=Snap&channel=0&user=admin&password=a1121C1123" + '&'+ d.getTime();
  //   $('#video-stream').css('background-image', 'url(' + url + ')');
  // },1000);

  d = new Date();
  var url = "http://192.168.1.107/cgi-bin/api.cgi?cmd=Snap&channel=0&user=root&password=123456" + '&'+ d.getTime();
  $('#video-stream').css('background-image', 'url(' + url + ')');

  //send request to IP Camera
  var url="http://192.168.1.107/cgi-bin/api.cgi?user=root&password=123456&cmd=PtzCtrl&token=141263da251ab9e";
  var json;
  $('#btn-up').click(function(e){
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Up", "speed": 32}}];
    makeRequest();
  });
  $('#btn-down').click(function(e){
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Down", "speed": 32}}];
    makeRequest();
  });
  $('#btn-left').click(function(e){
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Left", "speed": 32}}];
    makeRequest();
  });
  $('#btn-right').click(function(e){
    e.preventDefault();
    json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Right", "speed": 32}}];
    makeRequest();
  });
  $('#btn-test').click(function(e){
    e.preventDefault();
    json = [{"cmd":"PtzCtrl","action":0,"param":{"channel":0,"op":"ToPos","speed":32,"id":1}}];
    makeRequest();
  });

  function makeRequest(){
    //create request for shopify
    var req=new XMLHttpRequest();

    req.onreadystatechange=function(e) {
      if(req.readyState==4) {
        var showErrorTab=false;
        if(req.status==200) {
          console.log("response:"+req.responseText);
        } else {
          console.log("Error calling PtzCtrl");
        }
      }
    }

    req.open("POST",url);
    req.setRequestHeader("Accept","application/json");
    req.send(JSON.stringify(json));
  }




  // setInterval(function(){
  //   $('#video-stream').css('background-image', '');
  // },2000)
});
