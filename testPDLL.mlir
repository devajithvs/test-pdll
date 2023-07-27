module {
  func.func @basic_queries(%arg0: f32) -> f32 {
    %c2_i32 = arith.constant 2 : i32
    %0 = "hello.french"(%c2_i32) {bonjour = 1 : i32} : (i32) -> f32
    %1 = "hello.english"(%c2_i32) {hello = 1 : i32} : (i32) -> f32
    %2 = "hello.japanese"(%0, %1) {konnichiwa = 1 : i32} : (f32, f32) -> f32
    %3 = "hello.spanish"(%1, %2) {hola = 1 : i32} : (f32, f32) -> f32
    return %3 : f32
  }
}
