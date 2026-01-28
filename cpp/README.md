<br />
<p align="center">
  <h3 align="center">Rovers</h3>
  <p align="center">
    A modified prototype of <a href="https://github.com/gaurav-dixitv/rovers">Gaurav Dixit's Rovers environment</a>.
    <br />
    <br />
  </p>
</p>


<!-- TABLE OF CONTENTS -->
## Table of Contents

- [Table of Contents](#table-of-contents)
- [Rovers](#rovers)
  - [Built With](#built-with)
- [License](#license)



<!-- ABOUT THE PROJECT -->
## Rovers

A prototype of the Rovers environment. A C++ template header-only library. Directly use the headers in your project -- does not need compilation, packaging and installation. Python bindings are provided for prototyping.

The library is designed for
* Ease of use: Manages all the hairy bits (think memory, threading, generics).
* Flexibility: Algorithms are agent/environment/learning agnostic and can be mixed in a variety of ways. Language bindings provide an interface for rapid prototyping.
* Performance: This library runs the rover domain much faster than a pure python equivalent.

### Built With
The python bindings are generated using cppyy.
* [cppyy](https://cppyy.readthedocs.io/en/latest/)

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.