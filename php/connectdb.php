<?php
  $servername = 'localhost';
  $user = 'root';
  $pass = '';
  $db = 'camera_control';

  //create db connection
  $conn = new mysqli($servername, $user, $pass, $db);
  if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
  }

  function createTable($conn){
    //if table not exists create table
    $query = "SELECT id FROM camerapos";
    $result = mysqli_query($conn, $query);

    if(empty($result)) {
      $query = "CREATE TABLE camerapos (
                id VARCHAR(12) NOT NULL,
                toppos INT(2) NOT NULL,
                leftpos INT(2) NOT NULL,
                rightpos INT(2) NOT NULL,
                bottompos INT(2) NOT NULL,
                PRIMARY KEY  (id)
                )";
      $result =  mysqli_query($conn, $query);
    }
  }

  createTable($conn);
?>
