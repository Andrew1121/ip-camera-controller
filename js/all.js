$(function(){
  var req, left = 0, right = 0, top = 0, bottom = 0;
  var upDur;
  var initTimer = 0;
  d = new Date();
  var url = "http://192.168.1.107/cgi-bin/api.cgi?cmd=Snap&channel=0&user=root&password=123456" + '&'+ d.getTime();
  $('#video-stream').css('background-image', 'url(' + url + ')');

  //send request to IP Camera
  var ctrlUrl="http://192.168.1.107/cgi-bin/api.cgi?user=root&password=123456&cmd=PtzCtrl&token=141263da251ab9e";
  var json;
  var initHandler = {
    init: function(){
      $(".loading-overlay").show();
      makeCtrlRequest("RightDown");
      var init = setInterval(function(){
        initTimer++;
        if(initTimer == 22){
          clearInterval(init);
          makeCtrlRequest("Stop");
          $(".loading-overlay").hide();
        }
      }, 1000);
    }
  }
  var controlHandler = {
    init: function(){
      $('#btn-up').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Up");
        upDur = setInterval(function(){
          if(top == 21){
            clearInterval(upDur);
            $(this).trigger('mouseup');
          }
          top++;
        }, 1000);
      }).on('mouseup', function() {
        makeCtrlRequest("Stop");
        updateRecord();
      });
      $('#btn-down').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Down");
        downDur = setInterval(function(){
          if(bottom == 21){
            clearInterval(upDur);
            $(this).trigger('mouseup');
          }
          bottom++;
        }, 1000);
      }).on('mouseup', function() {
        makeCtrlRequest("Stop");
        updateRecord();
      });
      $('#btn-left').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Left");
        leftDur = setInterval(function(){
          if(left == 17){
            clearInterval(upDur);
            $(this).trigger('mouseup');
          }
          left++;
        }, 1000);
      }).on('mouseup', function() {
        makeCtrlRequest("Stop");
        updateRecord();
      });
      $('#btn-right').on('mousedown', function(e) {
        e.preventDefault();
        makeCtrlRequest("Right");
        rightDur = setInterval(function(){
          if(right == 17){
            clearInterval(upDur);
            $(this).trigger('mouseup');
          }
          right++;
        }, 1000);
      }).on('mouseup', function() {
        makeCtrlRequest("Stop");
        updateRecord();
      });
      $('#btn-zoom1').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","100%");
      });
      $('#btn-zoom15').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","150%");
      });
      $('#btn-zoom20').click(function(e) {
        e.preventDefault();
        $("#video-stream").css("background-size","200%");
      });
    }
  }

  // $('#btn-test').click(function(e){
  //   e.preventDefault();
  //   json = [{"cmd":"PtzCtrl","action":0,"param":{"channel":0,"op":"ToPos","speed":32,"id":1}}];
  //   makeCtrlRequest();
  // });

  function makeCtrlRequest(param){
    //create request for shopify
    if(param=="Stop"){
      json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": "Stop"}}];
    }else{
      json = [{"cmd": "PtzCtrl", "action": 0, "param": {"channel": 0, "op": param, "speed": 32}}];
      console.log(json);
    }

    req = new XMLHttpRequest();

    req.onreadystatechange=function() {
      if(req.readyState==4) {
        var showErrorTab=false;
      }
    }

    req.open("POST",ctrlUrl);
    req.setRequestHeader("Accept","application/json");
    req.send(JSON.stringify(json));
  }

  function updateRecord(){
    var data = {id:'currentpos', top:'"+ top +"', left:'"+ left +"', right:'"+ right +"', bottom:'"+ bottom +"'};
    console.log(data);
    $.post('../php/updatedb.php', data, function(returnedData){
      console.log(returnedData);
    });
  }

  // initHandler.init();
  controlHandler.init();
});
