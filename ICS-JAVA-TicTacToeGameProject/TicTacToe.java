import java.util.Scanner;

public class TicTacToe {
    // initialize game board
    private static char[][] board = {
            {'1', '2', '3'},
            {'4', '5', '6'},
            {'7', '8', '9'}
    };

    // player names
    private static String player1 = "Player 1";
    private static String player2 = "Player 2";

    // initialize game variables
    private static String currentPlayer = player1; // player 1 starts
    private static int moves = 0;
    private static int consecutiveIncorrectMoves = 0; // tracks consecutive incorrect moves
    private static int totalIncorrectMovesPlayer1 = 0; // tracks total incorrect moves for player 1
    private static int totalIncorrectMovesPlayer2 = 0; // tracks total incorrect moves for player 2
    private static boolean gameOver = false;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int mode = -1; // initialize mode with an invalid value

        // loops until a valid int is selected (Must pick from 0 1 or 2)
        while (mode != 0 && mode != 1 && mode != 2) {
            System.out.println("Enter 1 for 2 players, 2 for playing against the CPU or 0 to quit.");
            if (scanner.hasNextInt()) { //checks to see if input is int 
                mode = scanner.nextInt();
            } else {
                System.out.println("Invalid input. Please enter 0, 1, or 2."); //if not an int, will say it is invalid and try again.
                scanner.next(); // clears invalid input from the scanner
            }
        }

        if (mode == 0) {
            System.out.println("Exiting the game.");
        }

        printBoard();

        while (!gameOver) {
            if (mode == 1 || (mode == 2 && currentPlayer.equals(player1))) { //if it is mode 1 and it is player 1's turn
                playerMove(scanner);
            } else {
                cpuMove(); // otherwise it is the CPU's turn 
            }

            printBoard();

            if (checkWin()) {
                System.out.println("Game Over! " + currentPlayer + " Wins.");
                gameOver = true;
            } else if (moves == 9) { //if CheckWin() isnt satisfied after 9 moves obviously tie (all tiles filled)
                System.out.println("Game Over! It is a tie.");
                gameOver = true;
            }

            switchPlayer();
        }
    }

    // method for switching players
    private static void switchPlayer() {
        if (currentPlayer.equals(player1)) {
            currentPlayer = player2;
        } else {
            currentPlayer = player1;
        }
    }

    //  method for handling player's move
    private static void playerMove(Scanner scanner) {
    System.out.println(currentPlayer + " turn:");
    int choice = scanner.nextInt();
    if (choice == 0) {
        System.out.println(currentPlayer + " forfeit the game.");
        gameOver = true;
        return; // Return immediately after game over
    }
    if (!isValidMove(choice)) {
        System.out.println("Incorrect entry, please try again.");
        consecutiveIncorrectMoves++; // counter for incorrect moves 
        // checks if consecutive incorrect moves reach the limit of 3
        if (consecutiveIncorrectMoves == 3) {
            System.out.println(currentPlayer + " forfeit the game due to 3 consecutive incorrect moves.");
            gameOver = true;
            return; // Return immediately after game over
        } else {
            // increment total incorrect moves for the current player
            if (currentPlayer.equals(player1)) {
                totalIncorrectMovesPlayer1++;
            } else {
                totalIncorrectMovesPlayer2++;
            }
        }
    } else {
        consecutiveIncorrectMoves = 0; // reset consecutive incorrect move counter only when a valid move is made
        makeMove(choice);
    }
}



    // method for making a move on the board
    private static void makeMove(int choice) {
        char symbol = ' '; // initialize symbol
        if (currentPlayer.equals(player1)) { //if it is player 1's turn, the char that player 1 will deploy is X 
            symbol = 'X';
        } else {
            symbol = 'O'; //otherwise it is player 2's choice to deploy O
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (board[i][j] == (char)(choice + '0')) {
                    board[i][j] = symbol; //X or O will replace the tile on the tictactoe board based on matrix index
                    moves++;
                }
            }
        }
    }

    // method for checking if a move is valid
    private static boolean isValidMove(int choice) {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                char c = board[i][j];
                if (c == (char)(choice + '0')) {  // if we add 0 to the input (i.e 1) and it equals the char in the board then it is a valid choice  (1+0=1)
                    return true;
                }
            }
        }
        return false;
    }

    // CPU's move method 
    private static void cpuMove() {
    System.out.println("CPU is thinking...");

    // first check for winning move
    for (int i = 1; i <= 9; i++) {
        if (isValidMove(i)) {
            makeMove(i); // potentially make a move for the cpu
            if (checkWin()) {
                System.out.println("CPU wins!"); // message indicating the CPU wins
            } else {
                board[(i - 1) / 3][(i - 1) % 3] = (char)(i + '0'); // undo the move if it doesn't result in a win (replace tile back to the int that it was)
            }
        }
    }
    // otherwise make a random valid move
    int randomMove;
    do {
        randomMove = (int)(Math.random() * 9) + 1; // generate a random move (1 to 9)
    } while (!isValidMove(randomMove)); // repeat until a valid move is found

    makeMove(randomMove); // Make the random move for the current player
}


    // method for printing the current state of the board
    private static void printBoard() {
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                System.out.print(board[i][j] + " ");
                if (j < 2) {  // j<2 because we want to print a | after each number in each row but not in the last column
                    System.out.print("| ");
                }
            }
            System.out.println();
            if (i < 2) { // i<2 because we want to print this separator line after each row but not in the last row
                System.out.println("- + - + -");
            }
        }
    }

    // method for checking win conditions
    private static boolean checkWin() {
        for (int i = 0; i < 3; i++) {
            if (board[i][0] == board[i][1] && board[i][1] == board[i][2]) { //all horizontal
                return true;
            }
            if (board[0][i] == board[1][i] && board[1][i] == board[2][i]) { //all vertical
                return true; 
            }
        }
        if (board[0][0] == board[1][1] && board[1][1] == board[2][2]) { //diagonal case top left to bot right
            return true;
        }
        return board[0][2] == board[1][1] && board[1][1] == board[2][0]; //2nd diagonal case top right to bot left
    }
}
